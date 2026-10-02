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

# lệch tối đa so với median hàng xóm
MAX_FLOW_DEVIATION = 1.5

# cần ít nhất 2 hàng xóm để check
MIN_NEIGHBORS = 2


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

def create_blocks(
    master,
    edge_mask,
    text_mask
):

    h, w = master.shape[:2]

    blocks = []

    index = 0

    total_pixels = (
        BLOCK_SIZE * BLOCK_SIZE
    )

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

            # ----------------------------
            # EDGE
            # ----------------------------

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


            # ----------------------------
            # TEXT
            # ----------------------------

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


            # ----------------------------
            # BLOCK HỢP LỆ
            # ----------------------------

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

def compute_block_flow(
    master_gray,
    test_gray,
    blocks
):

    h, w = master_gray.shape[:2]

    results = []


    for block in blocks:

        x = block["x"]
        y = block["y"]


        # block không đủ cấu trúc
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


        # ----------------------------
        # vùng tìm kiếm
        # ----------------------------

        sx1 = max(
            0,
            x - MAX_DISP
        )

        sy1 = max(
            0,
            y - MAX_DISP
        )

        sx2 = min(
            w,
            x + BLOCK_SIZE + MAX_DISP
        )

        sy2 = min(
            h,
            y + BLOCK_SIZE + MAX_DISP
        )


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


        # ----------------------------
        # NCC
        # ----------------------------

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

    # map block theo row / col
    flow_map = {}

    for r in flow_results:

        if not r["valid"]:
            continue

        col = r["x"] // BLOCK_SIZE
        row = r["y"] // BLOCK_SIZE

        flow_map[
            (row, col)
        ] = r


    checked_results = []


    for r in flow_results:

        result = r.copy()


        # block đã invalid từ trước
        if not r["valid"]:

            result["sanity_valid"] = False

            result["sanity_reason"] = (
                r["reason"]
            )

            result["neighbor_count"] = 0

            result["median_dx"] = np.nan
            result["median_dy"] = np.nan

            result["deviation"] = np.nan

            checked_results.append(
                result
            )

            continue


        col = r["x"] // BLOCK_SIZE
        row = r["y"] // BLOCK_SIZE


        neighbor_dx = []
        neighbor_dy = []


        # ========================================
        # LẤY HÀNG XÓM
        # ========================================

        for dy_grid in range(
            -SANITY_RADIUS,
            SANITY_RADIUS + 1
        ):

            for dx_grid in range(
                -SANITY_RADIUS,
                SANITY_RADIUS + 1
            ):

                # bỏ chính block hiện tại
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


        # ========================================
        # QUÁ ÍT HÀNG XÓM
        # ========================================

        if (
            neighbor_count
            < MIN_NEIGHBORS
        ):

            result["sanity_valid"] = True

            result["sanity_reason"] = (
                "FEW_NEIGHBORS"
            )

            result["neighbor_count"] = (
                neighbor_count
            )

            result["median_dx"] = np.nan
            result["median_dy"] = np.nan

            result["deviation"] = 0.0


            checked_results.append(
                result
            )

            continue


        # ========================================
        # MEDIAN HÀNG XÓM
        # ========================================

        median_dx = float(
            np.median(
                neighbor_dx
            )
        )

        median_dy = float(
            np.median(
                neighbor_dy
            )
        )


        # ========================================
        # ĐỘ LỆCH VECTOR
        # ========================================

        deviation = float(

            np.sqrt(

                (
                    r["dx"]
                    - median_dx
                ) ** 2

                +

                (
                    r["dy"]
                    - median_dy
                ) ** 2

            )

        )


        # ========================================
        # CHECK
        # ========================================

        sanity_valid = (
            deviation
            <= MAX_FLOW_DEVIATION
        )


        result["sanity_valid"] = (
            sanity_valid
        )


        if sanity_valid:

            result["sanity_reason"] = (
                "OK"
            )

        else:

            result["sanity_reason"] = (
                "FLOW_OUTLIER"
            )


        result["neighbor_count"] = (
            neighbor_count
        )

        result["median_dx"] = (
            median_dx
        )

        result["median_dy"] = (
            median_dy
        )

        result["deviation"] = (
            deviation
        )


        checked_results.append(
            result
        )


    return checked_results


# ============================================================
# MAIN
# ============================================================

master = cv2.imread(
    MASTER_PATH
)

test = cv2.imread(
    TEST_PATH
)


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

print(
    f"score = {ecc_score:.4f}"
)

print(
    f"dx = {warp_matrix[0, 2]:.2f}"
)

print(
    f"dy = {warp_matrix[1, 2]:.2f}"
)

print(
    f"angle = {angle:.3f}"
)


# ============================================================
# TEXT + EDGE MASK
# ============================================================

text_mask = create_text_mask(
    master_gray
)


edge_mask, gradient_mag = (
    create_edge_mask(
        master_gray
    )
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

print(
    f"Tổng block       : "
    f"{len(blocks)}"
)

print(
    f"Block được giữ   : "
    f"{len(valid_blocks)}"
)

print(
    f"Block bị loại    : "
    f"{len(blocks) - len(valid_blocks)}"
)

print(
    f"MIN_EDGE_RATIO   : "
    f"{MIN_EDGE_RATIO}"
)

print(
    f"MIN_TEXT_RATIO   : "
    f"{MIN_TEXT_RATIO}"
)


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


# ============================================================
# FLOW STATS
# ============================================================

print("\n===== FLOW STATS =====")

print(
    f"conf min    = "
    f"{confs.min():.4f}"
)

print(
    f"conf mean   = "
    f"{confs.mean():.4f}"
)

print(
    f"conf median = "
    f"{np.median(confs):.4f}"
)

print(
    f"conf max    = "
    f"{confs.max():.4f}"
)


print(
    f"dx min/max  = "
    f"{dxs.min()} / {dxs.max()}"
)

print(
    f"dy min/max  = "
    f"{dys.min()} / {dys.max()}"
)


print(
    f"conf >=0.95 : "
    f"{np.sum(confs >= 0.95)}"
)

print(
    f"0.85-0.95   : "
    f"{np.sum((confs >= 0.85) & (confs < 0.95))}"
)

print(
    f"conf <0.85  : "
    f"{np.sum(confs < 0.85)}"
)


# ============================================================
# SANITY CHECK
# ============================================================

sanity_results = (
    sanity_check_blocks(
        flow_results
    )
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


print(
    "\n===== SANITY CHECK ====="
)

print(
    f"Flow trước sanity : "
    f"{len(valid_flows)}"
)

print(
    f"Flow được giữ     : "
    f"{len(sanity_good)}"
)

print(
    f"Flow bị loại      : "
    f"{len(sanity_bad)}"
)

print(
    f"MAX DEVIATION     : "
    f"{MAX_FLOW_DEVIATION}"
)


# ============================================================
# IN OUTLIER
# ============================================================

if len(sanity_bad) > 0:

    print(
        "\n===== SANITY OUTLIER ====="
    )


    for r in sanity_bad[:30]:

        print(

            f"B{r['index']:03d} | "

            f"dx={r['dx']:2d} "
            f"dy={r['dy']:2d} | "

            f"median=("
            f"{r['median_dx']:.1f},"
            f"{r['median_dy']:.1f}) | "

            f"dev="
            f"{r['deviation']:.2f} | "

            f"neighbors="
            f"{r['neighbor_count']}"

        )


# ============================================================
# FLOW SAMPLE
# ============================================================

print(
    "\n===== FLOW SAMPLE ====="
)

count = 0


for r in flow_results:

    if not r["valid"]:
        continue


    print(

        f"B{r['index']:03d} "

        f"edge="
        f"{r['edge_ratio']:.3f} "

        f"text="
        f"{r['text_ratio']:.3f} "

        f"dx="
        f"{r['dx']:2d} "

        f"dy="
        f"{r['dy']:2d} "

        f"conf="
        f"{r['confidence']:.3f}"

    )


    count += 1


    if count >= 30:
        break


# ============================================================
# VISUAL BLOCK
# ============================================================

block_vis = (
    master_canvas.copy()
)


for b in blocks:

    x = b["x"]
    y = b["y"]


    if b["valid_structure"]:

        color = (
            0,
            255,
            0
        )

    else:

        color = (
            0,
            0,
            255
        )


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

flow_vis = (
    master_canvas.copy()
)


for r in flow_results:

    if not r["valid"]:
        continue


    x = r["x"]
    y = r["y"]

    dx = r["dx"]
    dy = r["dy"]

    conf = r["confidence"]


    cx = (
        x
        + BLOCK_SIZE // 2
    )

    cy = (
        y
        + BLOCK_SIZE // 2
    )


    ex = int(
        cx
        + dx * ARROW_SCALE
    )

    ey = int(
        cy
        + dy * ARROW_SCALE
    )


    if conf >= 0.95:

        color = (
            0,
            255,
            0
        )

    elif conf >= 0.85:

        color = (
            0,
            255,
            255
        )

    else:

        color = (
            0,
            0,
            255
        )


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
# VISUAL SANITY BLOCK
# ============================================================

sanity_vis = (
    master_canvas.copy()
)


for r in sanity_results:

    if not r["valid"]:
        continue


    x = r["x"]
    y = r["y"]


    if r["sanity_valid"]:

        # xanh = giữ
        color = (
            0,
            255,
            0
        )

    else:

        # đỏ = outlier
        color = (
            0,
            0,
            255
        )


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

sanity_flow_vis = (
    master_canvas.copy()
)


for r in sanity_results:

    if not r["valid"]:
        continue


    x = r["x"]
    y = r["y"]


    cx = (
        x
        + BLOCK_SIZE // 2
    )

    cy = (
        y
        + BLOCK_SIZE // 2
    )


    # --------------------------------------------
    # block bị sanity loại
    # --------------------------------------------

    if not r["sanity_valid"]:

        cv2.line(
            sanity_flow_vis,
            (
                cx - 5,
                cy - 5
            ),
            (
                cx + 5,
                cy + 5
            ),
            (
                0,
                0,
                255
            ),
            2
        )

        cv2.line(
            sanity_flow_vis,
            (
                cx + 5,
                cy - 5
            ),
            (
                cx - 5,
                cy + 5
            ),
            (
                0,
                0,
                255
            ),
            2
        )

        continue


    dx = r["dx"]
    dy = r["dy"]


    ex = int(
        cx
        + dx * ARROW_SCALE
    )

    ey = int(
        cy
        + dy * ARROW_SCALE
    )


    color = (
        0,
        255,
        0
    )


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
# HIỂN THỊ BLOCK
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
    "GREEN = đủ EDGE + TEXT | RED = INVALID"
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# FLOW TRƯỚC SANITY
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
    "SANITY CHECK | "
    "GREEN = KEEP | "
    "RED = FLOW OUTLIER"
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# FLOW SAU SANITY
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
    "FLOW AFTER SANITY | "
    "GREEN = KEEP | "
    "RED X = OUTLIER"
)

plt.axis("off")

plt.tight_layout()


plt.show()