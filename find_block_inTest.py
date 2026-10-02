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

# Ít nhất 2% pixel trong block phải là cạnh chữ
MIN_EDGE_RATIO = 0.02

# Ít nhất 4% pixel trong block phải thuộc chữ đen
MIN_TEXT_RATIO = 0.04

ARROW_SCALE = 5


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
        y:y+h,
        x:x+w
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
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
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
            cv2.INTER_LINEAR |
            cv2.WARP_INVERSE_MAP
        ),
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned_test, warp_matrix, cc


# ============================================================
# TẠO TEXT MASK
# ============================================================

def create_text_mask(master_gray):
    """
    Chữ đen -> trắng (255)
    Nền     -> đen  (0)
    """

    blurred = cv2.GaussianBlur(
        master_gray,
        (3, 3),
        0
    )

    _, text_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return text_mask


# ============================================================
# TẠO EDGE MASK
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
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    edge_mask = cv2.dilate(
        edge_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return edge_mask, magnitude_8u


# ============================================================
# CHIA BLOCK + KIỂM TRA EDGE + TEXT
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
# NCC - TÌM BLOCK TRONG TEST
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

        # Block không đủ chữ/cạnh -> bỏ
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
# MAIN
# ============================================================

master = cv2.imread(MASTER_PATH)
test = cv2.imread(TEST_PATH)

if master is None:
    raise ValueError("Không đọc được Master")

if test is None:
    raise ValueError("Không đọc được Test")


print("Master:", master.shape[:2])
print("Test  :", test.shape[:2])


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
    f"dx = {warp_matrix[0,2]:.2f}"
)

print(
    f"dy = {warp_matrix[1,2]:.2f}"
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

print(
    f"Tổng block       : {len(blocks)}"
)

print(
    f"Block được giữ   : {len(valid_blocks)}"
)

print(
    f"Block bị loại    : "
    f"{len(blocks) - len(valid_blocks)}"
)

print(
    f"MIN_EDGE_RATIO   : {MIN_EDGE_RATIO}"
)

print(
    f"MIN_TEXT_RATIO   : {MIN_TEXT_RATIO}"
)


# ============================================================
# NCC FLOW
# ============================================================

flow_results = compute_block_flow(
    master_gray,
    aligned_gray,
    blocks
)

valid_flows = [r for r in flow_results if r["valid"]]

confs = np.array([r["confidence"] for r in valid_flows])
dxs = np.array([r["dx"] for r in valid_flows])
dys = np.array([r["dy"] for r in valid_flows])

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
# IN SAMPLE
# ============================================================

print("\n===== FLOW SAMPLE =====")

count = 0

for r in flow_results:

    if not r["valid"]:
        continue

    print(
        f"B{r['index']:03d} "
        f"edge={r['edge_ratio']:.3f} "
        f"text={r['text_ratio']:.3f} "
        f"dx={r['dx']:2d} "
        f"dy={r['dy']:2d} "
        f"conf={r['confidence']:.3f}"
    )

    count += 1

    if count >= 30:
        break


# ============================================================
# VISUAL BLOCK
# ============================================================

block_vis = master_canvas.copy()


for b in blocks:

    x = b["x"]
    y = b["y"]

    if b["valid_structure"]:
        color = (0, 255, 0)    # xanh
    else:
        color = (0, 0, 255)    # đỏ

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
# VISUAL FLOW
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


    if dx == 0 and dy == 0:

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
# HIỂN THỊ
# ============================================================

plt.figure(figsize=(18, 7))

plt.subplot(2, 1, 1)

plt.imshow(
    text_mask,
    cmap="gray"
)

plt.title(
    "TEXT MASK - trắng = chữ"
)

plt.axis("off")


plt.subplot(2, 1, 2)

plt.imshow(
    edge_mask,
    cmap="gray"
)

plt.title(
    "EDGE MASK - trắng = cạnh chữ"
)

plt.axis("off")

plt.tight_layout()


plt.figure(figsize=(20, 6))

plt.imshow(
    cv2.cvtColor(
        block_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "GREEN = đủ EDGE + đủ TEXT | "
    "RED = INVALID"
)

plt.axis("off")

plt.tight_layout()


plt.figure(figsize=(20, 6))

plt.imshow(
    cv2.cvtColor(
        flow_vis,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"FLOW | block={BLOCK_SIZE} | "
    f"max_disp=±{MAX_DISP}"
)

plt.axis("off")

plt.tight_layout()


plt.show()