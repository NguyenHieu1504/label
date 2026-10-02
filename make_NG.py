import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

# 1 ảnh OK
IMAGE_PATH = Path(
    r"D:\Folder_Labelimg\img_label\WIN_20260818_10_52_55_Pro.jpg"
)

# Folder lưu 10 ảnh NG
OUTPUT_DIR = Path(
    r"D:\check_textLabel\Make_NG"
)


# ============================================================
# ROI CHO PHÉP TẠO LỖI
# (x, y, w, h)
# ============================================================

ROI = (
    2350,   # x
    360,   # y
    1213,   # width
    302,   # height
)


# ============================================================
# TẠO BAO NHIÊU ẢNH NG TỪ 1 ẢNH OK
# ============================================================

NG_PER_OK = 10


# ============================================================
# THÔNG SỐ FAKE DEFECT
# ============================================================

CUT_LENGTH_RANGE = (10, 25)

FADE_PROBABILITY = 0.30
FADE_ALPHA_RANGE = (0.55, 0.90)

MIN_CHANGED_PIXELS = 8

# Khoảng cách tối thiểu giữa vị trí lỗi của các ảnh NG.
# Tăng số này nếu muốn 10 lỗi nằm cách xa nhau hơn.
MIN_CENTER_DISTANCE = 15

SEED = 42


# ============================================================
# READ IMAGE
# ============================================================

def read_image(path):
    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(path, image):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ok, data = cv2.imencode(
        path.suffix,
        image
    )

    if not ok:
        raise RuntimeError(
            f"Không lưu được: {path}"
        )

    data.tofile(str(path))


# ============================================================
# EXTRACT TEXT MASK
# ============================================================

def extract_text_mask(image):
    """
    Chữ tối -> 255
    Nền sáng -> 0
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )


    # ========================================================
    # BỎ NHIỄU NHỎ
    # ========================================================

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            8
        )
    )

    cleaned = np.zeros_like(
        binary
    )

    for i in range(1, n):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= 3:

            cleaned[
                labels == i
            ] = 255

    return cleaned


# ============================================================
# SKELETON
# ============================================================

def skeletonize(binary):

    work = binary.copy()

    skeleton = np.zeros_like(
        work
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3)
    )

    while cv2.countNonZero(work) > 0:

        eroded = cv2.erode(
            work,
            kernel
        )

        opened = cv2.dilate(
            eroded,
            kernel
        )

        skeleton |= cv2.subtract(
            work,
            opened
        )

        work = eroded

    return skeleton


# ============================================================
# GET CANDIDATE POINTS
# ============================================================

def get_candidate_points(skeleton):

    sk = (
        skeleton > 0
    ).astype(np.uint8)

    neighbor_count = cv2.filter2D(
        sk,
        cv2.CV_16S,
        np.ones(
            (3, 3),
            np.uint8
        )
    )

    # Ưu tiên điểm nằm giữa nét
    candidate = (
        (sk == 1)
        & (neighbor_count >= 3)
        & (neighbor_count <= 5)
    )

    ys, xs = np.where(
        candidate
    )

    # Nếu quá ít candidate
    if len(xs) < 10:

        ys, xs = np.where(
            sk == 1
        )

    return np.column_stack(
        (xs, ys)
    )


# ============================================================
# ESTIMATE STROKE DIRECTION
# ============================================================

def estimate_stroke_direction(
    skeleton,
    cx,
    cy,
    radius=12
):

    h, w = skeleton.shape

    x1 = max(
        0,
        cx - radius
    )

    x2 = min(
        w,
        cx + radius + 1
    )

    y1 = max(
        0,
        cy - radius
    )

    y2 = min(
        h,
        cy + radius + 1
    )


    ys, xs = np.where(
        skeleton[
            y1:y2,
            x1:x2
        ] > 0
    )


    if len(xs) < 3:

        angle = random.uniform(
            0,
            np.pi
        )

        return (
            np.cos(angle),
            np.sin(angle)
        )


    points = np.column_stack(
        (
            xs + x1 - cx,
            ys + y1 - cy
        )
    ).astype(np.float32)


    covariance = np.cov(
        points,
        rowvar=False
    )

    values, vectors = np.linalg.eigh(
        covariance
    )

    dx, dy = vectors[
        :,
        np.argmax(values)
    ]


    norm = max(
        np.hypot(dx, dy),
        1e-6
    )

    return (
        float(dx / norm),
        float(dy / norm)
    )


# ============================================================
# KIỂM TRA CENTER CÓ QUÁ GẦN LỖI CŨ KHÔNG
# ============================================================

def center_is_far_enough(
    cx,
    cy,
    used_centers
):

    for old_x, old_y in used_centers:

        distance = np.hypot(
            cx - old_x,
            cy - old_y
        )

        if distance < MIN_CENTER_DISTANCE:
            return False

    return True


# ============================================================
# TẠO 1 MISSING STROKE
# ============================================================

def add_missing_stroke(
    roi_image,
    used_centers
):

    text_mask = extract_text_mask(
        roi_image
    )

    skeleton = skeletonize(
        text_mask
    )

    points = get_candidate_points(
        skeleton
    )


    if len(points) == 0:
        return None


    # ========================================================
    # RANDOM THỨ TỰ CANDIDATE
    # ========================================================

    points = points.copy()

    np.random.shuffle(
        points
    )


    # ========================================================
    # DISTANCE TRANSFORM
    # dùng để tính độ dày nét
    # ========================================================

    distance_map = cv2.distanceTransform(
        (
            text_mask > 0
        ).astype(np.uint8),
        cv2.DIST_L2,
        5
    )


    h, w = text_mask.shape


    # ========================================================
    # THỬ CÁC VỊ TRÍ
    # ========================================================

    for cx, cy in points:

        cx = int(cx)
        cy = int(cy)


        # tránh sát biên ROI
        if (
            cx < 3
            or cy < 3
            or cx >= w - 3
            or cy >= h - 3
        ):
            continue


        # ====================================================
        # KIỂM TRA KHÔNG TRÙNG CÁC LỖI TRƯỚC
        # ====================================================

        if not center_is_far_enough(
            cx,
            cy,
            used_centers
        ):
            continue


        # ====================================================
        # HƯỚNG NÉT
        # ====================================================

        dx, dy = estimate_stroke_direction(
            skeleton,
            cx,
            cy
        )


        # ====================================================
        # RANDOM XOAY NHẸ
        # ====================================================

        angle = np.deg2rad(
            random.uniform(
                -12,
                12
            )
        )


        new_dx = (
            dx * np.cos(angle)
            - dy * np.sin(angle)
        )

        new_dy = (
            dx * np.sin(angle)
            + dy * np.cos(angle)
        )

        dx = new_dx
        dy = new_dy


        # ====================================================
        # LENGTH
        # ====================================================

        cut_length = random.randint(
            *CUT_LENGTH_RANGE
        )


        # ====================================================
        # ĐỘ DÀY NÉT
        # ====================================================

        stroke_width = max(
            1,
            int(
                round(
                    2
                    * distance_map[
                        cy,
                        cx
                    ]
                )
            )
        )


        cut_thickness = (
            stroke_width
            + random.randint(
                1,
                3
            )
        )


        # ====================================================
        # TẠO LINE CẮT
        # ====================================================

        half = cut_length / 2


        p1 = (
            int(
                round(
                    cx
                    - dx * half
                )
            ),
            int(
                round(
                    cy
                    - dy * half
                )
            )
        )


        p2 = (
            int(
                round(
                    cx
                    + dx * half
                )
            ),
            int(
                round(
                    cy
                    + dy * half
                )
            )
        )


        cut_mask = np.zeros(
            (h, w),
            np.uint8
        )


        cv2.line(
            cut_mask,
            p1,
            p2,
            255,
            cut_thickness,
            cv2.LINE_AA
        )


        # ====================================================
        # CHỈ CẮT PHẦN GIAO VỚI CHỮ
        # ====================================================

        defect_mask = cv2.bitwise_and(
            cut_mask,
            text_mask
        )


        changed_pixels = cv2.countNonZero(
            defect_mask
        )


        if (
            changed_pixels
            < MIN_CHANGED_PIXELS
        ):
            continue


        # ====================================================
        # DILATE NHẸ
        # ====================================================

        defect_mask = cv2.dilate(
            defect_mask,
            np.ones(
                (3, 3),
                np.uint8
            ),
            iterations=1
        )


        # ====================================================
        # TÁI TẠO BACKGROUND
        # ====================================================

        background = cv2.inpaint(
            roi_image,
            defect_mask,
            3,
            cv2.INPAINT_TELEA
        )


        output = roi_image.copy()

        selected = (
            defect_mask > 0
        )


        # ====================================================
        # FADE / HARD
        # ====================================================

        if (
            random.random()
            < FADE_PROBABILITY
        ):

            defect_type = "fade"

            alpha = random.uniform(
                *FADE_ALPHA_RANGE
            )


            faded = cv2.addWeighted(
                roi_image,
                1.0 - alpha,
                background,
                alpha,
                0
            )


            output[selected] = (
                faded[selected]
            )

        else:

            defect_type = "hard"

            output[selected] = (
                background[selected]
            )


        info = {
            "type": defect_type,
            "center": (cx, cy),
            "length": cut_length,
            "thickness": cut_thickness,
            "changed_pixels": changed_pixels,
        }


        return (
            output,
            info
        )


    return None


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )


    # ========================================================
    # CREATE OUTPUT FOLDER
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # READ ORIGINAL IMAGE
    # ========================================================

    image = read_image(
        IMAGE_PATH
    )


    if image is None:

        print(
            "Không đọc được ảnh:",
            IMAGE_PATH
        )

        return


    image_h, image_w = (
        image.shape[:2]
    )


    print(
        f"Image size: "
        f"{image_w} x {image_h}"
    )


    # ========================================================
    # ROI
    # ========================================================

    x, y, w, h = ROI


    x = max(
        0,
        x
    )

    y = max(
        0,
        y
    )


    x2 = min(
        image_w,
        x + w
    )

    y2 = min(
        image_h,
        y + h
    )


    if (
        x >= x2
        or y >= y2
    ):

        print(
            "ROI không hợp lệ:",
            ROI
        )

        return


    print(
        f"ROI: "
        f"x={x}, "
        f"y={y}, "
        f"w={x2-x}, "
        f"h={y2-y}"
    )


    # ========================================================
    # ORIGINAL ROI
    # ========================================================

    original_roi = image[
        y:y2,
        x:x2
    ].copy()


    # ========================================================
    # DANH SÁCH CENTER ĐÃ DÙNG
    # ========================================================

    used_centers = []


    created = 0


    # ========================================================
    # CREATE 10 NG IMAGES
    # ========================================================

    for ng_index in range(
        1,
        NG_PER_OK + 1
    ):


        # Luôn bắt đầu từ ảnh ROI OK ban đầu
        result = add_missing_stroke(
            original_roi,
            used_centers
        )


        if result is None:

            print(
                f"[FAIL] Không tìm được "
                f"vị trí cho NG {ng_index}"
            )

            continue


        fake_roi, info = result


        # ====================================================
        # GHI NHỚ CENTER
        # ====================================================

        cx, cy = info[
            "center"
        ]

        used_centers.append(
            (cx, cy)
        )


        # ====================================================
        # COPY FULL IMAGE
        # ====================================================

        output = image.copy()


        # ====================================================
        # CHỈ THAY ROI
        # ====================================================

        output[
            y:y2,
            x:x2
        ] = fake_roi


        # ====================================================
        # TỌA ĐỘ TRÊN FULL IMAGE
        # ====================================================

        full_center = (
            x + cx,
            y + cy
        )


        # ====================================================
        # OUTPUT NAME
        # ====================================================

        output_name = (
            f"{IMAGE_PATH.stem}"
            f"_ng_{ng_index:02d}.png"
        )


        output_path = (
            OUTPUT_DIR
            / output_name
        )


        # ====================================================
        # SAVE
        # ====================================================

        save_image(
            output_path,
            output
        )


        created += 1


        print(
            f"[OK] {output_name}"
        )

        print(
            f"     type   = "
            f"{info['type']}"
        )

        print(
            f"     center ROI  = "
            f"{info['center']}"
        )

        print(
            f"     center FULL = "
            f"{full_center}"
        )

        print(
            f"     length = "
            f"{info['length']}"
        )

        print(
            f"     thickness = "
            f"{info['thickness']}"
        )


    # ========================================================
    # RESULT
    # ========================================================

    print(
        "\n============================="
    )

    print(
        "HOÀN THÀNH"
    )

    print(
        "============================="
    )

    print(
        f"Ảnh OK đầu vào : "
        f"{IMAGE_PATH.name}"
    )

    print(
        f"Số ảnh NG      : "
        f"{created}"
    )

    print(
        f"Output folder  : "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()