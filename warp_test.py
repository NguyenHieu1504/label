import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# ============================================================
# PATH
# ============================================================

MASTER_PATH = r"D:\dataset_label\Anomalib\data_anomalib\img_line\train\good\master.jpg"
TEST_PATH   = r"D:\dataset_label\Anomalib\data_anomalib\img_line\train\good\test_1.jpg"


# ============================================================
# THAM SỐ
# ============================================================

BLOCK_SIZE = 32
MAX_DISP = 3

MIN_EDGE_RATIO = 0.02
MIN_TEXT_RATIO = 0.04

ARROW_SCALE = 5


# ============================================================
# SANITY CHECK
# ============================================================

SANITY_RADIUS = 1
MAX_FLOW_DEVIATION = 1.5
MIN_NEIGHBORS = 2


# ============================================================
# FILL INVALID BLOCK
# ============================================================

FILL_RADIUS = 1
MIN_FILL_NEIGHBORS = 2
MAX_FILL_ITER = 20


# ============================================================
# DENSE FLOW
# ============================================================

DENSE_SMOOTH_SIGMA = 3.0


# ============================================================
# LẤY MÀU NỀN
# ============================================================

def get_border_color(image, border=10):
    pixels = np.concatenate([
        image[:border].reshape(-1, 3),
        image[-border:].reshape(-1, 3),
        image[:, :border].reshape(-1, 3),
        image[:, -border:].reshape(-1, 3),
    ])

    color = np.median(
        pixels,
        axis=0
    ).astype(np.uint8)

    return tuple(color.tolist())


# ============================================================
# PADDING
# ============================================================

def pad_to_canvas(image, target_h, target_w):
    h, w = image.shape[:2]

    bg_color = get_border_color(image)

    canvas = np.full(
        (target_h, target_w, 3),
        bg_color,
        dtype=np.uint8
    )

    x = (target_w - w) // 2
    y = (target_h - h) // 2

    canvas[
        y:y + h,
        x:x + w
    ] = image

    return canvas


# ============================================================
# ECC
# ============================================================

def ecc_align(master, test):
    master_gray = cv2.cvtColor(
        master,
        cv2.COLOR_BGR2GRAY
    )

    test_gray = cv2.cvtColor(
        test,
        cv2.COLOR_BGR2GRAY
    )

    master_float = (
        master_gray.astype(np.float32)
        / 255.0
    )

    test_float = (
        test_gray.astype(np.float32)
        / 255.0
    )

    master_float = cv2.GaussianBlur(
        master_float,
        (5, 5),
        0
    )

    test_float = cv2.GaussianBlur(
        test_float,
        (5, 5),
        0
    )

    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-7
    )

    cc, warp_matrix = cv2.findTransformECC(
        master_float,
        test_float,
        warp_matrix,
        cv2.MOTION_EUCLIDEAN,
        criteria
    )

    h, w = master.shape[:2]

    aligned_test = cv2.warpAffine(
        test,
        warp_matrix,
        (w, h),
        flags=(
            cv2.INTER_LINEAR
            | cv2.WARP_INVERSE_MAP
        ),
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned_test, warp_matrix, cc


# ============================================================
# TEXT MASK
# ============================================================

def create_text_mask(master_gray):
    blurred = cv2.GaussianBlur(
        master_gray,
        (3, 3),
        0
    )

    _, text_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    return text_mask


# ============================================================
# EDGE MASK
# ============================================================

def create_edge_mask(master_gray):
    blurred = cv2.GaussianBlur(
        master_gray,
        (3, 3),
        0
    )

    gx = cv2.Scharr(
        blurred,
        cv2.CV_32F,
        1,
        0
    )

    gy = cv2.Scharr(
        blurred,
        cv2.CV_32F,
        0,
        1
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude_8u = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, edge_mask = cv2.threshold(
        magnitude_8u,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    edge_mask = cv2.dilate(
        edge_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return edge_mask, magnitude_8u


# ============================================================
# CHIA BLOCK + LỌC EDGE/TEXT
# ============================================================

def create_blocks(master, edge_mask, text_mask):
    h, w = master.shape[:2]

    blocks = []
    index = 0

    total_pixels = BLOCK_SIZE * BLOCK_SIZE

    for y in range(
        0,
        h - BLOCK_SIZE + 1,
        BLOCK_SIZE
    ):
        for x in range(
            0,
            w - BLOCK_SIZE + 1,
            BLOCK_SIZE
        ):
            edge_block = edge_mask[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ]

            edge_pixels = cv2.countNonZero(
                edge_block
            )

            edge_ratio = (
                edge_pixels / total_pixels
            )

            text_block = text_mask[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ]

            text_pixels = cv2.countNonZero(
                text_block
            )

            text_ratio = (
                text_pixels / total_pixels
            )

            valid_structure = (
                edge_ratio >= MIN_EDGE_RATIO
                and
                text_ratio >= MIN_TEXT_RATIO
            )

            blocks.append({
                "index": index,
                "x": x,
                "y": y,
                "edge_ratio": edge_ratio,
                "text_ratio": text_ratio,
                "valid_structure": valid_structure
            })

            index += 1

    return blocks


# ============================================================
# NCC - TÌM DX DY
# ============================================================

def compute_block_flow(master_gray, test_gray, blocks):
    h, w = master_gray.shape[:2]

    results = []

    for block in blocks:
        x = block["x"]
        y = block["y"]

        if not block["valid_structure"]:
            results.append({
                **block,
                "dx": 0,
                "dy": 0,
                "confidence": -1.0,
                "valid": False,
                "reason": "NO_STRUCTURE"
            })
            continue

        template = master_gray[
            y:y + BLOCK_SIZE,
            x:x + BLOCK_SIZE
        ]

        sx1 = max(0, x - MAX_DISP)
        sy1 = max(0, y - MAX_DISP)
        sx2 = min(w, x + BLOCK_SIZE + MAX_DISP)
        sy2 = min(h, y + BLOCK_SIZE + MAX_DISP)

        search = test_gray[
            sy1:sy2,
            sx1:sx2
        ]

        if (
            search.shape[0] < BLOCK_SIZE
            or
            search.shape[1] < BLOCK_SIZE
        ):
            results.append({
                **block,
                "dx": 0,
                "dy": 0,
                "confidence": -1.0,
                "valid": False,
                "reason": "SEARCH_SMALL"
            })
            continue

        response = cv2.matchTemplate(
            search,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, confidence, _, max_loc = (
            cv2.minMaxLoc(response)
        )

        best_x = sx1 + max_loc[0]
        best_y = sy1 + max_loc[1]

        dx = best_x - x
        dy = best_y - y

        results.append({
            **block,
            "dx": int(dx),
            "dy": int(dy),
            "confidence": float(confidence),
            "valid": True,
            "reason": "OK"
        })

    return results


# ============================================================
# SANITY CHECK BLOCK
# ============================================================

def sanity_check_blocks(flow_results):
    flow_map = {}

    for r in flow_results:
        if not r["valid"]:
            continue

        col = r["x"] // BLOCK_SIZE
        row = r["y"] // BLOCK_SIZE

        flow_map[(row, col)] = r

    checked_results = []

    for r in flow_results:
        result = r.copy()

        if not r["valid"]:
            result["sanity_valid"] = False
            result["sanity_reason"] = r["reason"]
            result["neighbor_count"] = 0
            result["median_dx"] = np.nan
            result["median_dy"] = np.nan
            result["deviation"] = np.nan

            checked_results.append(result)
            continue

        col = r["x"] // BLOCK_SIZE
        row = r["y"] // BLOCK_SIZE

        neighbor_dx = []
        neighbor_dy = []

        for dy_grid in range(
            -SANITY_RADIUS,
            SANITY_RADIUS + 1
        ):
            for dx_grid in range(
                -SANITY_RADIUS,
                SANITY_RADIUS + 1
            ):
                if (
                    dx_grid == 0
                    and
                    dy_grid == 0
                ):
                    continue

                neighbor = flow_map.get(
                    (
                        row + dy_grid,
                        col + dx_grid
                    )
                )

                if neighbor is None:
                    continue

                neighbor_dx.append(
                    neighbor["dx"]
                )

                neighbor_dy.append(
                    neighbor["dy"]
                )

        neighbor_count = len(
            neighbor_dx
        )

        if (
            neighbor_count
            < MIN_NEIGHBORS
        ):
            result["sanity_valid"] = True
            result["sanity_reason"] = "FEW_NEIGHBORS"
            result["neighbor_count"] = neighbor_count
            result["median_dx"] = np.nan
            result["median_dy"] = np.nan
            result["deviation"] = 0.0

            checked_results.append(result)
            continue

        median_dx = float(
            np.median(neighbor_dx)
        )

        median_dy = float(
            np.median(neighbor_dy)
        )

        deviation = float(
            np.sqrt(
                (r["dx"] - median_dx) ** 2
                +
                (r["dy"] - median_dy) ** 2
            )
        )

        sanity_valid = (
            deviation
            <= MAX_FLOW_DEVIATION
        )

        result["sanity_valid"] = sanity_valid

        if sanity_valid:
            result["sanity_reason"] = "OK"
        else:
            result["sanity_reason"] = "FLOW_OUTLIER"

        result["neighbor_count"] = neighbor_count
        result["median_dx"] = median_dx
        result["median_dy"] = median_dy
        result["deviation"] = deviation

        checked_results.append(result)

    return checked_results


# ============================================================
# FILL INVALID BLOCK
# ============================================================

def fill_invalid_blocks(sanity_results):
    grid = {}

    # Khởi tạo grid
    for r in sanity_results:
        item = r.copy()

        row = r["y"] // BLOCK_SIZE
        col = r["x"] // BLOCK_SIZE

        if (
            r["valid"]
            and
            r["sanity_valid"]
        ):
            item["fill_valid"] = True
            item["dx_filled"] = float(r["dx"])
            item["dy_filled"] = float(r["dy"])
            item["fill_source"] = "ORIGINAL"
            item["fill_neighbor_count"] = 0
            item["fill_iteration"] = 0

        else:
            item["fill_valid"] = False
            item["dx_filled"] = np.nan
            item["dy_filled"] = np.nan
            item["fill_source"] = "INVALID"
            item["fill_neighbor_count"] = 0
            item["fill_iteration"] = -1

        grid[(row, col)] = item

    # Fill nhiều vòng để lan flow vào vùng invalid lớn
    for iteration in range(
        1,
        MAX_FILL_ITER + 1
    ):
        updates = []

        for (row, col), r in grid.items():
            if r["fill_valid"]:
                continue

            neighbor_dx = []
            neighbor_dy = []

            for dy_grid in range(
                -FILL_RADIUS,
                FILL_RADIUS + 1
            ):
                for dx_grid in range(
                    -FILL_RADIUS,
                    FILL_RADIUS + 1
                ):
                    if (
                        dx_grid == 0
                        and
                        dy_grid == 0
                    ):
                        continue

                    neighbor = grid.get(
                        (
                            row + dy_grid,
                            col + dx_grid
                        )
                    )

                    if neighbor is None:
                        continue

                    if not neighbor["fill_valid"]:
                        continue

                    neighbor_dx.append(
                        neighbor["dx_filled"]
                    )

                    neighbor_dy.append(
                        neighbor["dy_filled"]
                    )

            if (
                len(neighbor_dx)
                < MIN_FILL_NEIGHBORS
            ):
                continue

            fill_dx = float(
                np.median(neighbor_dx)
            )

            fill_dy = float(
                np.median(neighbor_dy)
            )

            updates.append(
                (
                    row,
                    col,
                    fill_dx,
                    fill_dy,
                    len(neighbor_dx),
                    iteration
                )
            )

        if len(updates) == 0:
            break

        # Apply đồng thời sau mỗi vòng
        for (
            row,
            col,
            fill_dx,
            fill_dy,
            neighbor_count,
            fill_iteration
        ) in updates:
            grid[(row, col)]["fill_valid"] = True
            grid[(row, col)]["dx_filled"] = fill_dx
            grid[(row, col)]["dy_filled"] = fill_dy
            grid[(row, col)]["fill_source"] = "FILLED"
            grid[(row, col)]["fill_neighbor_count"] = neighbor_count
            grid[(row, col)]["fill_iteration"] = fill_iteration

    results = list(
        grid.values()
    )

    results.sort(
        key=lambda r: r["index"]
    )

    return results


# ============================================================
# DENSIFY FIELD
# block flow -> pixel flow
# ============================================================

def densify_field(
    filled_results,
    image_h,
    image_w
):
    # --------------------------------------------------------
    # kiểm tra còn block invalid không
    # --------------------------------------------------------
    invalid_count = sum(
        not r["fill_valid"]
        for r in filled_results
    )

    if invalid_count > 0:
        raise RuntimeError(
            f"Còn {invalid_count} block chưa có flow."
        )

    # --------------------------------------------------------
    # số hàng / cột block
    # --------------------------------------------------------
    rows = (
        max(
            r["y"] // BLOCK_SIZE
            for r in filled_results
        )
        + 1
    )

    cols = (
        max(
            r["x"] // BLOCK_SIZE
            for r in filled_results
        )
        + 1
    )

    # --------------------------------------------------------
    # flow dạng coarse grid
    # mỗi ô tương ứng 1 block 32x32
    # --------------------------------------------------------
    coarse_dx = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    coarse_dy = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for r in filled_results:
        row = r["y"] // BLOCK_SIZE
        col = r["x"] // BLOCK_SIZE

        coarse_dx[row, col] = r["dx_filled"]
        coarse_dy[row, col] = r["dy_filled"]

    # --------------------------------------------------------
    # vùng mà grid block đang phủ
    # --------------------------------------------------------
    covered_h = rows * BLOCK_SIZE
    covered_w = cols * BLOCK_SIZE

    # --------------------------------------------------------
    # nội suy block flow -> pixel flow
    # --------------------------------------------------------
    dense_dx = cv2.resize(
        coarse_dx,
        (
            covered_w,
            covered_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    dense_dy = cv2.resize(
        coarse_dy,
        (
            covered_w,
            covered_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # nếu kích thước ảnh không chia hết BLOCK_SIZE
    # ví dụ 322x1273 thì block phủ 320x1248
    # phần dư sẽ lấy flow ở biên gần nhất
    # --------------------------------------------------------
    pad_bottom = max(
        0,
        image_h - covered_h
    )

    pad_right = max(
        0,
        image_w - covered_w
    )

    dense_dx = cv2.copyMakeBorder(
        dense_dx,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_REPLICATE
    )

    dense_dy = cv2.copyMakeBorder(
        dense_dy,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_REPLICATE
    )

    dense_dx = dense_dx[
        :image_h,
        :image_w
    ]

    dense_dy = dense_dy[
        :image_h,
        :image_w
    ]

    # --------------------------------------------------------
    # làm mượt trường flow
    # --------------------------------------------------------
    if DENSE_SMOOTH_SIGMA > 0:
        dense_dx = cv2.GaussianBlur(
            dense_dx,
            (0, 0),
            DENSE_SMOOTH_SIGMA
        )

        dense_dy = cv2.GaussianBlur(
            dense_dy,
            (0, 0),
            DENSE_SMOOTH_SIGMA
        )

    # --------------------------------------------------------
    # cubic có thể overshoot nhẹ -> giới hạn về range coarse
    # --------------------------------------------------------
    min_dx = float(coarse_dx.min())
    max_dx = float(coarse_dx.max())
    min_dy = float(coarse_dy.min())
    max_dy = float(coarse_dy.max())

    dense_dx = np.clip(
        dense_dx,
        min_dx,
        max_dx
    )

    dense_dy = np.clip(
        dense_dy,
        min_dy,
        max_dy
    )

    return (
        dense_dx,
        dense_dy,
        coarse_dx,
        coarse_dy
    )


# ============================================================
# WARP TEST BẰNG DENSE FLOW
# ============================================================

def warp_with_dense_flow(
    aligned_test,
    dense_dx,
    dense_dy
):
    """
    dense_dx, dense_dy được định nghĩa theo chiều:

        Master (x, y)
            -> vị trí tương ứng trong Test
               (x + dx, y + dy)

    cv2.remap cần map từ pixel đầu ra -> vị trí lấy mẫu trong ảnh nguồn,
    nên dùng:

        map_x = x + dense_dx
        map_y = y + dense_dy
    """

    h, w = aligned_test.shape[:2]

    if dense_dx.shape != (h, w):
        raise ValueError(
            f"dense_dx shape {dense_dx.shape} != image shape {(h, w)}"
        )

    if dense_dy.shape != (h, w):
        raise ValueError(
            f"dense_dy shape {dense_dy.shape} != image shape {(h, w)}"
        )

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = (
        grid_x
        + dense_dx.astype(np.float32)
    )

    map_y = (
        grid_y
        + dense_dy.astype(np.float32)
    )

    warped_test = cv2.remap(
        aligned_test,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped_test


# ============================================================
# MAIN
# ============================================================

master = cv2.imread(MASTER_PATH)
test = cv2.imread(TEST_PATH)

if master is None:
    raise ValueError(
        "Không đọc được Master"
    )

if test is None:
    raise ValueError(
        "Không đọc được Test"
    )

print(
    "Master:",
    master.shape[:2]
)

print(
    "Test  :",
    test.shape[:2]
)


# ============================================================
# PADDING
# ============================================================

target_h = max(
    master.shape[0],
    test.shape[0]
)

target_w = max(
    master.shape[1],
    test.shape[1]
)

master_canvas = pad_to_canvas(
    master,
    target_h,
    target_w
)

test_canvas = pad_to_canvas(
    test,
    target_h,
    target_w
)


# ============================================================
# ECC
# ============================================================

aligned_test, warp_matrix, ecc_score = (
    ecc_align(
        master_canvas,
        test_canvas
    )
)

master_gray = cv2.cvtColor(
    master_canvas,
    cv2.COLOR_BGR2GRAY
)

aligned_gray = cv2.cvtColor(
    aligned_test,
    cv2.COLOR_BGR2GRAY
)

angle = math.degrees(
    math.atan2(
        warp_matrix[1, 0],
        warp_matrix[0, 0]
    )
)

print("\n===== ECC =====")
print(f"score = {ecc_score:.4f}")
print(f"dx = {warp_matrix[0, 2]:.2f}")
print(f"dy = {warp_matrix[1, 2]:.2f}")
print(f"angle = {angle:.3f}")


# ============================================================
# TEXT + EDGE MASK
# ============================================================

text_mask = create_text_mask(
    master_gray
)

edge_mask, gradient_mag = create_edge_mask(
    master_gray
)


# ============================================================
# BLOCK
# ============================================================

blocks = create_blocks(
    master_canvas,
    edge_mask,
    text_mask
)

valid_blocks = [
    b
    for b in blocks
    if b["valid_structure"]
]

print("\n===== BLOCK =====")
print(f"Tổng block       : {len(blocks)}")
print(f"Block được giữ   : {len(valid_blocks)}")
print(f"Block bị loại    : {len(blocks) - len(valid_blocks)}")
print(f"MIN_EDGE_RATIO   : {MIN_EDGE_RATIO}")
print(f"MIN_TEXT_RATIO   : {MIN_TEXT_RATIO}")


# ============================================================
# NCC FLOW
# ============================================================

flow_results = compute_block_flow(
    master_gray,
    aligned_gray,
    blocks
)

valid_flows = [
    r
    for r in flow_results
    if r["valid"]
]

if len(valid_flows) == 0:
    raise RuntimeError(
        "Không có block hợp lệ sau NCC."
    )

confs = np.array([
    r["confidence"]
    for r in valid_flows
])

dxs = np.array([
    r["dx"]
    for r in valid_flows
])

dys = np.array([
    r["dy"]
    for r in valid_flows
])

print("\n===== FLOW STATS =====")
print(f"conf min    = {confs.min():.4f}")
print(f"conf mean   = {confs.mean():.4f}")
print(f"conf median = {np.median(confs):.4f}")
print(f"conf max    = {confs.max():.4f}")
print(f"dx min/max  = {dxs.min()} / {dxs.max()}")
print(f"dy min/max  = {dys.min()} / {dys.max()}")
print(f"conf >=0.95 : {np.sum(confs >= 0.95)}")
print(f"0.85-0.95   : {np.sum((confs >= 0.85) & (confs < 0.95))}")
print(f"conf <0.85  : {np.sum(confs < 0.85)}")


# ============================================================
# SANITY CHECK
# ============================================================

sanity_results = sanity_check_blocks(
    flow_results
)

sanity_good = [
    r
    for r in sanity_results
    if (
        r["valid"]
        and
        r["sanity_valid"]
    )
]

sanity_bad = [
    r
    for r in sanity_results
    if (
        r["valid"]
        and
        not r["sanity_valid"]
    )
]

print("\n===== SANITY CHECK =====")
print(f"Flow trước sanity : {len(valid_flows)}")
print(f"Flow được giữ     : {len(sanity_good)}")
print(f"Flow bị loại      : {len(sanity_bad)}")
print(f"MAX DEVIATION     : {MAX_FLOW_DEVIATION}")

if len(sanity_bad) > 0:
    print("\n===== SANITY OUTLIER =====")

    for r in sanity_bad[:30]:
        print(
            f"B{r['index']:03d} | "
            f"dx={r['dx']:2d} "
            f"dy={r['dy']:2d} | "
            f"median=("
            f"{r['median_dx']:.1f},"
            f"{r['median_dy']:.1f}) | "
            f"dev={r['deviation']:.2f} | "
            f"neighbors={r['neighbor_count']}"
        )


# ============================================================
# FILL INVALID
# ============================================================

filled_results = fill_invalid_blocks(
    sanity_results
)

original_count = sum(
    r["fill_source"] == "ORIGINAL"
    for r in filled_results
)

filled_count = sum(
    r["fill_source"] == "FILLED"
    for r in filled_results
)

remaining_invalid = sum(
    not r["fill_valid"]
    for r in filled_results
)

print("\n===== FILL INVALID =====")
print(f"Flow gốc hợp lệ   : {original_count}")
print(f"Block được fill   : {filled_count}")
print(f"Còn INVALID       : {remaining_invalid}")


# ============================================================
# IN SAMPLE BLOCK ĐƯỢC FILL
# ============================================================

print("\n===== FILLED SAMPLE =====")

count = 0

for r in filled_results:
    if r["fill_source"] != "FILLED":
        continue

    print(
        f"B{r['index']:03d} | "
        f"dx={r['dx_filled']:.1f} "
        f"dy={r['dy_filled']:.1f} | "
        f"neighbors={r['fill_neighbor_count']} | "
        f"iteration={r['fill_iteration']}"
    )

    count += 1

    if count >= 30:
        break


# ============================================================
# DENSE FLOW
# ============================================================

image_h, image_w = master_canvas.shape[:2]

(
    dense_dx,
    dense_dy,
    coarse_dx,
    coarse_dy
) = densify_field(
    filled_results,
    image_h,
    image_w
)

flow_magnitude = np.sqrt(
    dense_dx ** 2
    +
    dense_dy ** 2
)

print("\n===== DENSE FLOW =====")
print(f"Coarse flow shape : {coarse_dx.shape}")
print(f"Dense flow shape  : {dense_dx.shape}")
print(f"Dense dx min/max  : {dense_dx.min():.3f} / {dense_dx.max():.3f}")
print(f"Dense dy min/max  : {dense_dy.min():.3f} / {dense_dy.max():.3f}")
print(f"Magnitude min/max : {flow_magnitude.min():.3f} / {flow_magnitude.max():.3f}")


# ============================================================
# WARP TEST BẰNG DENSE FLOW
# ============================================================

warped_test = warp_with_dense_flow(
    aligned_test,
    dense_dx,
    dense_dy
)


# ============================================================
# DIFFERENCE TRƯỚC / SAU LOCAL WARP
# ============================================================

# Trước local warp: Master vs Test sau ECC
diff_before = cv2.absdiff(
    master_canvas,
    aligned_test
)

# Sau local warp: Master vs Test sau ECC + dense flow
diff_after = cv2.absdiff(
    master_canvas,
    warped_test
)


diff_before_gray = cv2.cvtColor(
    diff_before,
    cv2.COLOR_BGR2GRAY
)

diff_after_gray = cv2.cvtColor(
    diff_after,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# THỐNG KÊ WARP
# ============================================================

mean_diff_before = float(
    diff_before_gray.mean()
)

mean_diff_after = float(
    diff_after_gray.mean()
)


text_pixels = (
    text_mask > 0
)

if np.any(text_pixels):

    text_diff_before = float(
        diff_before_gray[
            text_pixels
        ].mean()
    )

    text_diff_after = float(
        diff_after_gray[
            text_pixels
        ].mean()
    )

else:

    text_diff_before = np.nan
    text_diff_after = np.nan


if mean_diff_before > 1e-9:

    global_improvement = (
        100.0
        * (
            mean_diff_before
            - mean_diff_after
        )
        / mean_diff_before
    )

else:

    global_improvement = 0.0


if (
    np.isfinite(text_diff_before)
    and text_diff_before > 1e-9
):

    text_improvement = (
        100.0
        * (
            text_diff_before
            - text_diff_after
        )
        / text_diff_before
    )

else:

    text_improvement = np.nan


print(
    "\n===== WARP RESULT ====="
)

print(
    f"Mean diff BEFORE warp : "
    f"{mean_diff_before:.3f}"
)

print(
    f"Mean diff AFTER warp  : "
    f"{mean_diff_after:.3f}"
)

print(
    f"Global improvement    : "
    f"{global_improvement:.2f}%"
)

print(
    f"Text diff BEFORE warp : "
    f"{text_diff_before:.3f}"
)

print(
    f"Text diff AFTER warp  : "
    f"{text_diff_after:.3f}"
)

print(
    f"Text improvement      : "
    f"{text_improvement:.2f}%"
)


# ============================================================
# VISUAL BLOCK
# ============================================================

block_vis = master_canvas.copy()

for b in blocks:
    x = b["x"]
    y = b["y"]

    if b["valid_structure"]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        block_vis,
        (x, y),
        (
            x + BLOCK_SIZE - 1,
            y + BLOCK_SIZE - 1
        ),
        color,
        1
    )


# ============================================================
# VISUAL FLOW TRƯỚC SANITY
# ============================================================

flow_vis = master_canvas.copy()

for r in flow_results:
    if not r["valid"]:
        continue

    x = r["x"]
    y = r["y"]

    dx = r["dx"]
    dy = r["dy"]

    conf = r["confidence"]

    cx = x + BLOCK_SIZE // 2
    cy = y + BLOCK_SIZE // 2

    ex = int(
        cx + dx * ARROW_SCALE
    )

    ey = int(
        cy + dy * ARROW_SCALE
    )

    if conf >= 0.95:
        color = (0, 255, 0)
    elif conf >= 0.85:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    if (
        dx == 0
        and
        dy == 0
    ):
        cv2.circle(
            flow_vis,
            (cx, cy),
            2,
            color,
            -1
        )
    else:
        cv2.arrowedLine(
            flow_vis,
            (cx, cy),
            (ex, ey),
            color,
            1,
            tipLength=0.35
        )


# ============================================================
# VISUAL SANITY
# ============================================================

sanity_vis = master_canvas.copy()

for r in sanity_results:
    if not r["valid"]:
        continue

    x = r["x"]
    y = r["y"]

    if r["sanity_valid"]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        sanity_vis,
        (x, y),
        (
            x + BLOCK_SIZE - 1,
            y + BLOCK_SIZE - 1
        ),
        color,
        2
    )


# ============================================================
# VISUAL FLOW SAU SANITY
# ============================================================

sanity_flow_vis = master_canvas.copy()

for r in sanity_results:
    if not r["valid"]:
        continue

    x = r["x"]
    y = r["y"]

    cx = x + BLOCK_SIZE // 2
    cy = y + BLOCK_SIZE // 2

    if not r["sanity_valid"]:
        cv2.line(
            sanity_flow_vis,
            (cx - 5, cy - 5),
            (cx + 5, cy + 5),
            (0, 0, 255),
            2
        )

        cv2.line(
            sanity_flow_vis,
            (cx + 5, cy - 5),
            (cx - 5, cy + 5),
            (0, 0, 255),
            2
        )

        continue

    dx = r["dx"]
    dy = r["dy"]

    ex = int(
        cx + dx * ARROW_SCALE
    )

    ey = int(
        cy + dy * ARROW_SCALE
    )

    color = (0, 255, 0)

    if (
        dx == 0
        and
        dy == 0
    ):
        cv2.circle(
            sanity_flow_vis,
            (cx, cy),
            2,
            color,
            -1
        )
    else:
        cv2.arrowedLine(
            sanity_flow_vis,
            (cx, cy),
            (ex, ey),
            color,
            1,
            tipLength=0.35
        )


# ============================================================
# VISUAL FILL INVALID
# ============================================================

fill_vis = master_canvas.copy()

for r in filled_results:
    x = r["x"]
    y = r["y"]

    if r["fill_source"] == "ORIGINAL":
        color = (0, 255, 0)      # xanh lá
    elif r["fill_source"] == "FILLED":
        color = (255, 0, 0)      # xanh dương
    else:
        color = (0, 0, 255)      # đỏ

    cv2.rectangle(
        fill_vis,
        (x, y),
        (
            x + BLOCK_SIZE - 1,
            y + BLOCK_SIZE - 1
        ),
        color,
        2
    )


# ============================================================
# VISUAL FLOW SAU FILL
# ============================================================

filled_flow_vis = master_canvas.copy()

for r in filled_results:
    if not r["fill_valid"]:
        continue

    x = r["x"]
    y = r["y"]

    cx = x + BLOCK_SIZE // 2
    cy = y + BLOCK_SIZE // 2

    dx = r["dx_filled"]
    dy = r["dy_filled"]

    ex = int(
        round(
            cx + dx * ARROW_SCALE
        )
    )

    ey = int(
        round(
            cy + dy * ARROW_SCALE
        )
    )

    if r["fill_source"] == "ORIGINAL":
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    if (
        abs(dx) < 1e-6
        and
        abs(dy) < 1e-6
    ):
        cv2.circle(
            filled_flow_vis,
            (cx, cy),
            2,
            color,
            -1
        )
    else:
        cv2.arrowedLine(
            filled_flow_vis,
            (cx, cy),
            (ex, ey),
            color,
            1,
            tipLength=0.35
        )


# ============================================================
# HIỂN THỊ TEXT + EDGE
# ============================================================

plt.figure(
    figsize=(18, 7)
)

plt.subplot(
    2,
    1,
    1
)

plt.imshow(
    text_mask,
    cmap="gray"
)

plt.title(
    "TEXT MASK - trắng = chữ"
)

plt.axis("off")

plt.subplot(
    2,
    1,
    2
)

plt.imshow(
    edge_mask,
    cmap="gray"
)

plt.title(
    "EDGE MASK - trắng = cạnh chữ"
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# BLOCK
# ============================================================

plt.figure(
    figsize=(20, 6)
)

plt.imshow(
    cv2.cvtColor(
        block_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "BLOCK | GREEN = đủ EDGE + TEXT | RED = INVALID"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# FLOW BEFORE SANITY
# ============================================================

plt.figure(
    figsize=(20, 6)
)

plt.imshow(
    cv2.cvtColor(
        flow_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"FLOW BEFORE SANITY | "
    f"block={BLOCK_SIZE} | "
    f"max_disp=±{MAX_DISP}"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# SANITY CHECK
# ============================================================

plt.figure(
    figsize=(20, 6)
)

plt.imshow(
    cv2.cvtColor(
        sanity_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "SANITY CHECK | GREEN = KEEP | RED = FLOW OUTLIER"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# FLOW AFTER SANITY
# ============================================================

plt.figure(
    figsize=(20, 6)
)

plt.imshow(
    cv2.cvtColor(
        sanity_flow_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "FLOW AFTER SANITY | GREEN = KEEP | RED X = OUTLIER"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# FILL INVALID
# ============================================================

plt.figure(
    figsize=(20, 6)
)

plt.imshow(
    cv2.cvtColor(
        fill_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "FILL INVALID | GREEN = ORIGINAL | BLUE = FILLED | RED = STILL INVALID"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# FLOW AFTER FILL
# ============================================================

plt.figure(
    figsize=(20, 6)
)

plt.imshow(
    cv2.cvtColor(
        filled_flow_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "FLOW AFTER FILL | GREEN = ORIGINAL | BLUE = FILLED"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# DENSE FLOW - DX
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    dense_dx,
    cmap="jet"
)

plt.colorbar(
    label="dx (pixel)"
)

plt.title(
    "DENSE FLOW - DX"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# DENSE FLOW - DY
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    dense_dy,
    cmap="jet"
)

plt.colorbar(
    label="dy (pixel)"
)

plt.title(
    "DENSE FLOW - DY"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# DENSE FLOW - MAGNITUDE
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    flow_magnitude,
    cmap="jet"
)

plt.colorbar(
    label="Displacement (pixel)"
)

plt.title(
    "DENSE FLOW - MAGNITUDE"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# TEST AFTER ECC
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    cv2.cvtColor(
        aligned_test,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "TEST AFTER ECC"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# TEST AFTER DENSE FLOW WARP
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    cv2.cvtColor(
        warped_test,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "TEST AFTER DENSE FLOW WARP"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# DIFFERENCE BEFORE LOCAL WARP
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    diff_before_gray,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"DIFFERENCE BEFORE LOCAL WARP | mean={mean_diff_before:.3f}"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# DIFFERENCE AFTER DENSE FLOW WARP
# ============================================================

plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    diff_after_gray,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"DIFFERENCE AFTER DENSE FLOW WARP | mean={mean_diff_after:.3f}"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# TEXT DIFFERENCE BEFORE / AFTER
# chỉ giữ vùng chữ Master để nhìn dễ hơn
# ============================================================

text_diff_before_vis = np.zeros_like(
    diff_before_gray
)

text_diff_after_vis = np.zeros_like(
    diff_after_gray
)

text_diff_before_vis[
    text_pixels
] = diff_before_gray[
    text_pixels
]

text_diff_after_vis[
    text_pixels
] = diff_after_gray[
    text_pixels
]


plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    text_diff_before_vis,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"TEXT DIFFERENCE BEFORE LOCAL WARP | mean={text_diff_before:.3f}"
)

plt.axis("off")
plt.tight_layout()


plt.figure(
    figsize=(20, 5)
)

plt.imshow(
    text_diff_after_vis,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"TEXT DIFFERENCE AFTER DENSE FLOW WARP | mean={text_diff_after:.3f}"
)

plt.axis("off")
plt.tight_layout()


plt.show()
