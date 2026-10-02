from pathlib import Path
import sys
import json
import base64

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from skimage.morphology import skeletonize



# ============================================================
# DESKEW CONFIG
# ============================================================
COLOR_DELTA_Y = 2.2
COLOR_DELTA_X = 2.2
SMOOTH_KERNEL = 31
MIN_GAP_SIZE = 8

EDGE_SEARCH = 30
EDGE_LOCAL_RADIUS = 22
SAMPLE_STEP = 5

ANGLE_TOLERANCE = 0.03
MAX_ROTATE_ITER = 3
BBOX_PADDING = 0


# ============================================================
# OBJECT DETECTION CONFIG
# ============================================================
COMPONENT_MIN_AREA = 2
MIN_ROI_SIZE = 10

# Normalize ánh sáng chỉ trong ROI người dùng vẽ
BACKGROUND_SIGMA_RATIO = 0.12
BACKGROUND_SIGMA_MIN = 7.0
BACKGROUND_SIGMA_MAX = 35.0

# Bbox nét đứt fit đúng object bên trong ROI, không nới thêm pixel

# TEXT MISSING CHECK
TOLERANCE_PX = 1.2
MIN_MISSING_LENGTH = 3

# OFFSET - tìm lại vị trí các ROI master trên ảnh mới
OFFSET_SEARCH_RADIUS = 100
OFFSET_MATCH_MIN_SCORE = 0.45

# IMAGE VIEW
ZOOM_STEP = 1.15
ZOOM_MIN = 1.0
ZOOM_MAX = 5.0

ROI_COLOR = "#ff9800"
OBJECT_FIT_COLOR = "#2196f3"
GOOD_COLOR = "#00c853"
NG_COLOR = "#ff1744"
TEXT_COLOR = "#ffffff"

# CONFIG DATA - file JSON được tạo cùng thư mục với script
CONFIG_PATH = Path(__file__).resolve().parent / "test_text_config.json"


# ============================================================
# COMMON
# ============================================================
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def config_exists():
    return CONFIG_PATH.exists()


def load_config():
    if not CONFIG_PATH.exists():
        return None

    try:
        return json.loads(
            CONFIG_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None


def save_config_file(data):
    temp_path = CONFIG_PATH.with_suffix(".json.tmp")

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temp_path.replace(CONFIG_PATH)
    return CONFIG_PATH


def encode_template_png(image):
    """Lưu ROI master trực tiếp vào JSON dưới dạng PNG base64."""
    if image is None or image.size == 0:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, buffer = cv2.imencode(".png", gray)

    if not ok:
        return ""

    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_template_png(text):
    if not text:
        return None

    try:
        data = base64.b64decode(text.encode("ascii"))
        array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def encode_gray_png(image):
    if image is None or image.size == 0:
        return ""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return ""

    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_gray_png(text):
    return decode_template_png(text)


def find_roi_offset(image, roi_config):
    """
    Tìm offset của đúng ROI này trên ảnh mới bằng template matching.
    Không dùng bbox label để tính offset.
    """
    template = decode_template_png(roi_config.get("template_png", ""))

    if template is None or template.size == 0:
        return None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    master_x = int(roi_config.get("x", 0))
    master_y = int(roi_config.get("y", 0))
    roi_w = int(roi_config.get("w", 0))
    roi_h = int(roi_config.get("h", 0))

    th, tw = template.shape[:2]

    if tw <= 0 or th <= 0:
        return None

    h_img, w_img = image_gray.shape[:2]

    search_x1 = max(0, master_x - OFFSET_SEARCH_RADIUS)
    search_y1 = max(0, master_y - OFFSET_SEARCH_RADIUS)

    search_x2 = min(
        w_img,
        master_x + roi_w + OFFSET_SEARCH_RADIUS
    )
    search_y2 = min(
        h_img,
        master_y + roi_h + OFFSET_SEARCH_RADIUS
    )

    search = image_gray[
        search_y1:search_y2,
        search_x1:search_x2
    ]

    if search.shape[1] < tw or search.shape[0] < th:
        return None

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_score, _, max_location = cv2.minMaxLoc(result)

    match_x = search_x1 + max_location[0]
    match_y = search_y1 + max_location[1]

    dx = match_x - master_x
    dy = match_y - master_y

    return {
        "dx": int(dx),
        "dy": int(dy),
        "score": float(max_score)
    }


# ============================================================
# DESKEW
# ============================================================
def smooth_profile(profile, kernel=31):
    if kernel % 2 == 0:
        kernel += 1
    profile = np.asarray(profile, dtype=np.float32)
    return cv2.GaussianBlur(profile.reshape(-1, 1), (1, kernel), 0).ravel()


def find_segments(mask, min_length=8):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    keep = (ends - starts + 1) >= min_length
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def detect_label(image):
    h_img, w_img = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]

    # TOP / BOTTOM
    x1, x2 = int(w_img * 0.12), int(w_img * 0.93)
    row = np.median(a[:, x1:x2], axis=1)
    row = smooth_profile(row, SMOOTH_KERNEL)

    center_color = np.median(row[int(h_img * 0.35):int(h_img * 0.65)])
    gaps = find_segments(row < center_color - COLOR_DELTA_Y, MIN_GAP_SIZE)

    cy = h_img // 2
    upper = [g for g in gaps if g[1] < cy]
    lower = [g for g in gaps if g[0] > cy]

    if not upper or not lower:
        raise RuntimeError("Không tìm được TOP/BOTTOM")

    top = max(upper, key=lambda g: g[1])[1] + 1
    bottom = min(lower, key=lambda g: g[0])[0] - 1

    # LEFT / RIGHT
    bbox_h = bottom - top
    y1 = top + int(bbox_h * 0.08)
    y2 = bottom - int(bbox_h * 0.08)

    col = np.median(a[y1:y2, :], axis=0)
    col = smooth_profile(col, SMOOTH_KERNEL)

    center_color = np.median(col[int(w_img * 0.35):int(w_img * 0.65)])
    gaps = find_segments(col < center_color - COLOR_DELTA_X, MIN_GAP_SIZE)

    cx = w_img // 2
    left_gaps = [g for g in gaps if g[1] < cx]
    right_gaps = [g for g in gaps if g[0] > cx]

    if not left_gaps or not right_gaps:
        raise RuntimeError("Không tìm được LEFT/RIGHT")

    left = max(left_gaps, key=lambda g: g[1])[1] + 1
    right = min(right_gaps, key=lambda g: g[0])[0] - 1
    return left, right, top, bottom


def find_transition(profile, expected_index, direction, radius=22):
    profile = np.asarray(profile, dtype=np.float32)
    gradient = np.diff(profile)
    if len(gradient) == 0:
        return None

    expected_index = int(round(expected_index))
    start = max(0, expected_index - radius)
    end = min(len(gradient), expected_index + radius + 1)
    if start >= end:
        return None

    local = gradient[start:end]
    score = local if direction > 0 else -local
    return start + int(np.argmax(score))


def find_edge_points(image, left, right, top, bottom):
    h_img, w_img = image.shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    a = cv2.GaussianBlur(a, (5, 5), 0)

    bbox_w = right - left
    bbox_h = bottom - top
    margin_x = max(10, int(bbox_w * 0.05))
    margin_y = max(10, int(bbox_h * 0.05))

    top_points, bottom_points = [], []
    left_points, right_points = [], []

    # TOP
    y_start = max(1, top - EDGE_SEARCH)
    y_end = min(h_img - 2, top + EDGE_SEARCH)
    expected = top - y_start

    for x in range(left + margin_x, right - margin_x, SAMPLE_STEP):
        profile = a[y_start:y_end + 1, x]
        idx = find_transition(profile, expected, direction=+1, radius=EDGE_LOCAL_RADIUS)
        if idx is not None:
            top_points.append([x, y_start + idx])

    # BOTTOM
    y_start = max(1, bottom - EDGE_SEARCH)
    y_end = min(h_img - 2, bottom + EDGE_SEARCH)
    expected = bottom - y_start

    for x in range(left + margin_x, right - margin_x, SAMPLE_STEP):
        profile = a[y_start:y_end + 1, x]
        idx = find_transition(profile, expected, direction=-1, radius=EDGE_LOCAL_RADIUS)
        if idx is not None:
            bottom_points.append([x, y_start + idx])

    # LEFT
    x_start = max(1, left - EDGE_SEARCH)
    x_end = min(w_img - 2, left + EDGE_SEARCH)
    expected = left - x_start

    for y in range(top + margin_y, bottom - margin_y, SAMPLE_STEP):
        profile = a[y, x_start:x_end + 1]
        idx = find_transition(profile, expected, direction=+1, radius=EDGE_LOCAL_RADIUS)
        if idx is not None:
            left_points.append([x_start + idx, y])

    # RIGHT
    x_start = max(1, right - EDGE_SEARCH)
    x_end = min(w_img - 2, right + EDGE_SEARCH)
    expected = right - x_start

    for y in range(top + margin_y, bottom - margin_y, SAMPLE_STEP):
        profile = a[y, x_start:x_end + 1]
        idx = find_transition(profile, expected, direction=-1, radius=EDGE_LOCAL_RADIUS)
        if idx is not None:
            right_points.append([x_start + idx, y])

    top_points = np.asarray(top_points, dtype=np.float32)
    bottom_points = np.asarray(bottom_points, dtype=np.float32)
    left_points = np.asarray(left_points, dtype=np.float32)
    right_points = np.asarray(right_points, dtype=np.float32)

    if len(top_points) < 10:
        raise RuntimeError("Không đủ điểm TOP")
    if len(bottom_points) < 10:
        raise RuntimeError("Không đủ điểm BOTTOM")
    if len(left_points) < 5:
        raise RuntimeError("Không đủ điểm LEFT")
    if len(right_points) < 5:
        raise RuntimeError("Không đủ điểm RIGHT")

    return top_points, bottom_points, left_points, right_points


def robust_fit_line(points, iterations=3):
    if len(points) < 10:
        raise RuntimeError("Không đủ điểm fit line")

    current = points.copy()

    for _ in range(iterations):
        line = cv2.fitLine(current, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(line[0][0]), float(line[1][0])
        x0, y0 = float(line[2][0]), float(line[3][0])

        x, y = current[:, 0], current[:, 1]
        distance = np.abs((x - x0) * vy - (y - y0) * vx) / np.hypot(vx, vy)

        median = np.median(distance)
        mad = np.median(np.abs(distance - median))
        threshold = max(1.5, median + 3.0 * mad)

        keep = distance <= threshold
        if np.count_nonzero(keep) < 10:
            break
        current = current[keep]

    line = cv2.fitLine(current, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(line[0][0]), float(line[1][0])
    x0, y0 = float(line[2][0]), float(line[3][0])
    return vx, vy, x0, y0, current


def get_edge_angle(points):
    vx, vy, _, _, inliers = robust_fit_line(points)
    angle = -np.degrees(np.arctan2(vy, vx))

    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180

    return angle, inliers


def get_label_angle(top_points, bottom_points):
    angle_top, top_inliers = get_edge_angle(top_points)
    angle_bottom, bottom_inliers = get_edge_angle(bottom_points)

    difference = abs(angle_top - angle_bottom)
    if difference > 1.0:
        print(f"WARNING: TOP/BOTTOM angle difference = {difference:.3f} deg")

    angle = (angle_top + angle_bottom) / 2.0
    return angle, angle_top, angle_bottom, top_inliers, bottom_inliers


def robust_median(values):
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    deviation = np.abs(values - median)
    mad = np.median(deviation)
    threshold = max(1.5, 3.0 * mad)
    good = values[deviation <= threshold]
    return float(median if len(good) == 0 else np.median(good))


def get_tight_bbox(top_points, bottom_points, left_points, right_points, image_shape):
    h_img, w_img = image_shape[:2]

    top = robust_median(top_points[:, 1])
    bottom = robust_median(bottom_points[:, 1])
    left = robust_median(left_points[:, 0])
    right = robust_median(right_points[:, 0])

    left = int(round(left)) - BBOX_PADDING
    right = int(round(right)) + BBOX_PADDING
    top = int(round(top)) - BBOX_PADDING
    bottom = int(round(bottom)) + BBOX_PADDING

    left = max(0, min(left, w_img - 1))
    right = max(0, min(right, w_img - 1))
    top = max(0, min(top, h_img - 1))
    bottom = max(0, min(bottom, h_img - 1))
    return left, right, top, bottom


def rotate_image(image, rotation_angle):
    h_img, w_img = image.shape[:2]
    center = (w_img / 2.0, h_img / 2.0)

    matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(np.ceil(h_img * sin + w_img * cos))
    new_h = int(np.ceil(h_img * cos + w_img * sin))

    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def analyze_label(image):
    left, right, top, bottom = detect_label(image)

    top_points, bottom_points, left_points, right_points = find_edge_points(
        image, left, right, top, bottom
    )

    angle, angle_top, angle_bottom, top_inliers, bottom_inliers = get_label_angle(
        top_points, bottom_points
    )

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "top_points": top_points,
        "bottom_points": bottom_points,
        "left_points": left_points,
        "right_points": right_points,
        "top_inliers": top_inliers,
        "bottom_inliers": bottom_inliers,
        "angle": angle,
        "angle_top": angle_top,
        "angle_bottom": angle_bottom,
    }


def deskew_to_zero(image):
    original = image
    total_correction = 0.0

    before = analyze_label(original)
    current = original

    for iteration in range(MAX_ROTATE_ITER):
        info = analyze_label(current)
        residual_angle = info["angle"]

        print(f"ITER {iteration + 1} | Residual={residual_angle:+.4f} deg")

        if abs(residual_angle) <= ANGLE_TOLERANCE:
            break

        total_correction += -residual_angle

        # Luôn rotate lại từ ảnh gốc để tránh interpolation nhiều lần
        current = rotate_image(original, total_correction)

    after = analyze_label(current)
    return current, before, after, total_correction


# ============================================================
# ROI OBJECT DETECTOR
# ============================================================

class ROIObjectDetector:
    def __init__(self):
        self.last_otsu_threshold = None
        self.last_normalized = None
        self.last_binary = None

    def preprocess_roi(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sigma = min(
            BACKGROUND_SIGMA_MAX,
            max(BACKGROUND_SIGMA_MIN, min(crop.shape[:2]) * BACKGROUND_SIGMA_RATIO)
        )

        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background = np.maximum(background, 1)
        normalized = cv2.divide(gray, background, scale=255)

        otsu_threshold, binary = cv2.threshold(
            normalized, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Bảo đảm nền trắng, object/chữ đen.
        if np.mean(binary == 0) > 0.50:
            binary = cv2.bitwise_not(binary)

        return normalized, binary, float(otsu_threshold)

    def detect(self, image, roi):
        if image is None:
            return None

        x, y, w, h = map(int, roi)
        image_h, image_w = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_w, x + w)
        y2 = min(image_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        normalized, binary, otsu_threshold = self.preprocess_roi(crop)
        mask = cv2.bitwise_not(binary)

        self.last_otsu_threshold = otsu_threshold
        self.last_normalized = normalized
        self.last_binary = binary

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        valid = []

        for i in range(1, count):
            cx, cy, cw, ch, area = stats[i]

            if area < COMPONENT_MIN_AREA or cw <= 0 or ch <= 0:
                continue

            valid.append((int(cx), int(cy), int(cw), int(ch)))

        if not valid:
            return None

        # Gom toàn bộ object/chữ nằm trong ROI thành MỘT bbox duy nhất.
        ox1 = min(cx for cx, cy, cw, ch in valid)
        oy1 = min(cy for cx, cy, cw, ch in valid)
        ox2 = max(cx + cw for cx, cy, cw, ch in valid)
        oy2 = max(cy + ch for cx, cy, cw, ch in valid)

        # Đưa bbox object về tọa độ ảnh.
        # Không nới thêm pixel: bbox fit đúng biên object detect được.
        bx1 = max(0, x1 + ox1)
        by1 = max(0, y1 + oy1)
        bx2 = min(image_w, x1 + ox2)
        by2 = min(image_h, y1 + oy2)

        if bx2 <= bx1 or by2 <= by1:
            return None

        return (
            int(bx1),
            int(by1),
            int(bx2 - bx1),
            int(by2 - by1)
        )


# ============================================================
# TEXT DEFECT CHECKER
# ============================================================

class TextDefectChecker:
    def __init__(self, tolerance_px=2.0, min_missing_length=6):
        self.tolerance_px = float(tolerance_px)
        self.min_missing_length = int(min_missing_length)
        self.last_master_binary = None

    @staticmethod
    def preprocess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        return binary

    def create_master_skeleton(self, master_crop):
        master_binary = self.preprocess(master_crop)
        self.last_master_binary = master_binary.copy()
        return skeletonize(master_binary > 0).astype(np.uint8) * 255

    def find_missing(self, master_skeleton, test_binary):
        # Mỗi pixel của distance chứa khoảng cách tới nét Test gần nhất.
        distance = cv2.distanceTransform(
            cv2.bitwise_not(test_binary),
            cv2.DIST_L2,
            5
        )

        missing = (
            (master_skeleton > 0)
            & (distance > self.tolerance_px)
        )

        return missing.astype(np.uint8) * 255, distance

    def analyze_missing(self, missing):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            missing,
            connectivity=8
        )

        filtered = np.zeros_like(missing)
        defects = []

        for i in range(1, count):
            length = int(stats[i, cv2.CC_STAT_AREA])
            if length < self.min_missing_length:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            filtered[labels == i] = 255
            defects.append({
                "length": length,
                "bbox": (x, y, w, h)
            })

        return filtered, defects

    @staticmethod
    def make_overlay(test_binary, master_skeleton):
        """
        Test Binary: trắng.
        Master Skeleton: đỏ.
        Dùng để nhìn trực tiếp skeleton Master có nằm đúng trên nét Test hay không.
        """
        overlay = cv2.cvtColor(test_binary, cv2.COLOR_GRAY2BGR)
        overlay[master_skeleton > 0] = (0, 0, 255)
        return overlay

    def check(self, test_crop, master_skeleton, master_binary=None):
        if (
            test_crop is None
            or test_crop.size == 0
            or master_skeleton is None
            or master_skeleton.size == 0
        ):
            return {
                "result": "NG",
                "defects": [],
                "master_binary": master_binary,
                "master_skeleton": master_skeleton,
                "test_binary": None,
                "overlay": None,
                "distance": None,
                "missing_raw": None,
                "missing": None,
                "missing_pixels": 0,
                "missing_ratio": 0.0,
                "error": "Invalid crop or master skeleton"
            }

        # 3. Test binary
        test_binary = self.preprocess(test_crop)

        # Đưa Test về cùng kích thước Master để có cùng hệ tọa độ pixel.
        size_mismatch_note = None
        master_h, master_w = master_skeleton.shape[:2]
        test_h, test_w = test_binary.shape[:2]

        if (test_h, test_w) != (master_h, master_w):
            size_mismatch_note = (
                f"Size mismatch resized: master={master_skeleton.shape[:2]} "
                f"test={test_binary.shape[:2]} -> resized to master size"
            )
            test_binary = cv2.resize(
                test_binary,
                (master_w, master_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 1. Master binary (debug).
        # Nếu config cũ không có Master binary thì dùng skeleton làm fallback
        # để cửa sổ debug vẫn hiển thị được.
        if master_binary is None:
            master_binary = master_skeleton.copy()
        else:
            master_binary = master_binary.copy()

        if master_binary.shape[:2] != (master_h, master_w):
            master_binary = cv2.resize(
                master_binary,
                (master_w, master_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 4. Overlay Master Skeleton lên Test Binary
        overlay = self.make_overlay(test_binary, master_skeleton)

        # 5. Distance map + missing raw
        missing_raw, distance = self.find_missing(
            master_skeleton,
            test_binary
        )

        # 6. Missing mask sau Connected Components filter
        missing_filtered, defects = self.analyze_missing(
            missing_raw
        )

        total_skeleton = cv2.countNonZero(master_skeleton)
        total_missing = cv2.countNonZero(missing_filtered)
        missing_ratio = total_missing / max(total_skeleton, 1)

        return {
            "result": "NG" if defects else "GOOD",
            "defects": defects,

            # 6 ảnh debug
            "master_binary": master_binary,
            "master_skeleton": master_skeleton.copy(),
            "test_binary": test_binary,
            "overlay": overlay,
            "distance": distance,
            "missing_raw": missing_raw,
            "missing": missing_filtered,

            "missing_pixels": int(total_missing),
            "missing_ratio": float(missing_ratio),
            "error": size_mismatch_note
        }


# ============================================================
# IMAGE CANVAS
# ============================================================
class ImageCanvas(QtWidgets.QLabel):
    roiFinished = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(900, 550)
        self.setStyleSheet("QLabel{background:#202020;border:1px solid #555;}")

        self.image = None
        self.original_pixmap = None
        self.image_draw_rect = None

        self.zoom_factor = 1.0
        self.pan_offset = QtCore.QPoint(0, 0)
        self.last_mouse_position = QtCore.QPoint()
        self.is_panning = False

        self.drawing = False
        self.draw_start = None
        self.draw_end = None

        self.regions = []

    def set_image(self, image):
        self.image = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, channels = rgb.shape
        qimg = QtGui.QImage(
            rgb.data, w_img, h_img, channels * w_img,
            QtGui.QImage.Format_RGB888
        ).copy()
        self.original_pixmap = QtGui.QPixmap.fromImage(qimg)

        self.regions = []
        self.drawing = False
        self.draw_start = None
        self.draw_end = None

        self.zoom_factor = 1.0
        self.pan_offset = QtCore.QPoint(0, 0)
        self.is_panning = False
        self.render()

    def set_regions(self, regions):
        self.regions = regions
        self.render()

    def clear_detection(self):
        self.regions = []
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.render()

    def cancel_draw(self):
        if not self.drawing:
            return
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.render()

    def draw_text_mask(self, painter, text_mask):
        if not text_mask:
            return

        mask = text_mask.get("mask")
        x = int(text_mask.get("x", 0))
        y = int(text_mask.get("y", 0))

        if mask is None or mask.size == 0:
            return

        h, w = mask.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask > 0] = (0, 200, 83, 255)

        qimage = QtGui.QImage(
            rgba.data,
            w,
            h,
            rgba.strides[0],
            QtGui.QImage.Format_RGBA8888
        ).copy()

        overlay = QtGui.QPixmap.fromImage(qimage)
        target = self.image_to_view_rect((x, y, w, h))

        painter.drawPixmap(
            target,
            overlay,
            QtCore.QRectF(0, 0, w, h)
        )

    def render(self):
        if self.original_pixmap is None:
            return

        view = self.contentsRect().size()
        target_size = QtCore.QSize(
            int(view.width() * self.zoom_factor),
            int(view.height() * self.zoom_factor)
        )
        pix = self.original_pixmap.scaled(
            target_size, QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.clamp_pan_offset(
            pix.width(), pix.height(),
            view.width(), view.height()
        )

        x = (view.width() - pix.width()) // 2 + self.pan_offset.x()
        y = (view.height() - pix.height()) // 2 + self.pan_offset.y()
        self.image_draw_rect = QtCore.QRect(x, y, pix.width(), pix.height())

        canvas = QtGui.QPixmap(view)
        canvas.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(canvas)
        painter.drawPixmap(x, y, pix)

        for region in self.regions:
            roi_rect = region["rect"]
            view_roi = self.image_to_view_rect(roi_rect)

            result = region.get("result")
            if result == "GOOD":
                roi_color = GOOD_COLOR
            elif result == "NG":
                roi_color = NG_COLOR
            else:
                roi_color = ROI_COLOR

            # ROI do người dùng vẽ: giữ nguyên bbox, vẽ nét liền.
            painter.setPen(QtGui.QPen(QtGui.QColor(roi_color), 2))
            painter.drawRect(view_roi)

            label = region["name"]
            if result:
                label += f" [{result}]"

            painter.setPen(QtGui.QPen(QtGui.QColor(TEXT_COLOR), 1))
            painter.drawText(
                view_roi.topLeft() + QtCore.QPointF(2, -4),
                label
            )

            # Bbox object tự fit bên trong ROI: nét đứt, không padding.
            object_rect = region.get("object_rect")
            if object_rect is not None:
                view_object = self.image_to_view_rect(object_rect)
                painter.setPen(
                    QtGui.QPen(
                        QtGui.QColor(OBJECT_FIT_COLOR),
                        2,
                        QtCore.Qt.DashLine
                    )
                )
                painter.drawRect(view_object)

            # Khi đã check, tô các pixel chữ còn tồn tại trên ảnh Test thành xanh.
            self.draw_text_mask(
                painter,
                region.get("text_mask")
            )

            # Chỉ khoanh đúng vùng NG thật sự từ missing mask.
            # Không khoanh ellipse lớn quanh cả chữ / cả ROI.
            self.draw_ng_contours(
                painter,
                region.get("ng_contours", [])
            )

        if self.drawing and self.draw_start is not None and self.draw_end is not None:
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(ROI_COLOR),
                    2,
                    QtCore.Qt.DashLine
                )
            )
            painter.drawRect(
                QtCore.QRect(self.draw_start, self.draw_end).normalized()
            )

        painter.end()
        self.setPixmap(canvas)

    def view_to_image_point(self, pos):
        if (
            self.image_draw_rect is None
            or self.image is None
            or not self.image_draw_rect.contains(pos)
        ):
            return None

        h_img, w_img = self.image.shape[:2]
        x = int(
            (pos.x() - self.image_draw_rect.x())
            * w_img / self.image_draw_rect.width()
        )
        y = int(
            (pos.y() - self.image_draw_rect.y())
            * h_img / self.image_draw_rect.height()
        )
        return (
            max(0, min(x, w_img - 1)),
            max(0, min(y, h_img - 1))
        )

    def view_rect_to_image_rect(self, rect):
        if self.image_draw_rect is None or self.image is None:
            return None

        rect = rect.intersected(self.image_draw_rect)
        if rect.isEmpty():
            return None

        h_img, w_img = self.image.shape[:2]
        x = int(
            (rect.x() - self.image_draw_rect.x())
            * w_img / self.image_draw_rect.width()
        )
        y = int(
            (rect.y() - self.image_draw_rect.y())
            * h_img / self.image_draw_rect.height()
        )
        w = int(rect.width() * w_img / self.image_draw_rect.width())
        h = int(rect.height() * h_img / self.image_draw_rect.height())
        return x, y, w, h

    def image_to_view_rect(self, rect):
        x, y, w, h = rect
        h_img, w_img = self.image.shape[:2]
        draw = self.image_draw_rect
        return QtCore.QRectF(
            draw.x() + x * draw.width() / w_img,
            draw.y() + y * draw.height() / h_img,
            w * draw.width() / w_img,
            h * draw.height() / h_img
        )

    def image_to_view_point(self, x, y):
        h_img, w_img = self.image.shape[:2]
        draw = self.image_draw_rect

        return QtCore.QPointF(
            draw.x() + float(x) * draw.width() / w_img,
            draw.y() + float(y) * draw.height() / h_img
        )

    def draw_ng_contours(self, painter, contours):
        if not contours:
            return

        painter.setPen(
            QtGui.QPen(
                QtGui.QColor(NG_COLOR),
                2,
                QtCore.Qt.SolidLine
            )
        )
        painter.setBrush(QtCore.Qt.NoBrush)

        for contour in contours:
            if contour is None or len(contour) < 2:
                continue

            polygon = QtGui.QPolygonF()

            for px, py in contour:
                polygon.append(
                    self.image_to_view_point(px, py)
                )

            if len(polygon) >= 3:
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)

    def mousePressEvent(self, event):
        if self.image is None:
            return

        if event.button() == QtCore.Qt.RightButton:
            self.is_panning = True
            self.last_mouse_position = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return

        if event.button() != QtCore.Qt.LeftButton:
            return

        if self.view_to_image_point(event.pos()) is None:
            return

        self.drawing = True
        self.draw_start = event.pos()
        self.draw_end = event.pos()
        self.setCursor(QtCore.Qt.CrossCursor)
        self.render()

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.last_mouse_position
            self.last_mouse_position = event.pos()
            self.pan_offset += delta
            self.render()
            return

        if self.drawing:
            self.draw_end = event.pos()
            self.render()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.is_panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            return

        if event.button() != QtCore.Qt.LeftButton or not self.drawing:
            return

        self.draw_end = event.pos()
        self.drawing = False
        self.setCursor(QtCore.Qt.ArrowCursor)

        view_rect = QtCore.QRect(
            self.draw_start, self.draw_end
        ).normalized()
        image_rect = self.view_rect_to_image_rect(view_rect)

        self.draw_start = None
        self.draw_end = None

        if (
            image_rect is None
            or image_rect[2] < MIN_ROI_SIZE
            or image_rect[3] < MIN_ROI_SIZE
        ):
            self.render()
            return

        self.render()
        self.roiFinished.emit(image_rect)

    def wheelEvent(self, event):
        if self.original_pixmap is None:
            return

        if event.angleDelta().y() > 0:
            self.zoom_factor = min(
                self.zoom_factor * ZOOM_STEP,
                ZOOM_MAX
            )
        else:
            self.zoom_factor = max(
                self.zoom_factor / ZOOM_STEP,
                ZOOM_MIN
            )

        if self.zoom_factor <= ZOOM_MIN:
            self.zoom_factor = ZOOM_MIN
            self.pan_offset = QtCore.QPoint(0, 0)

        self.render()

    def clamp_pan_offset(self, image_width, image_height, view_width, view_height):
        if image_width > view_width:
            max_x = (image_width - view_width) // 2
            x = max(-max_x, min(self.pan_offset.x(), max_x))
        else:
            x = 0

        if image_height > view_height:
            max_y = (image_height - view_height) // 2
            y = max(-max_y, min(self.pan_offset.y(), max_y))
        else:
            y = 0

        self.pan_offset = QtCore.QPoint(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render()


# ============================================================
# DEBUG WINDOW - 6 IMAGES
# ============================================================

class DebugImageWindow(QtWidgets.QDialog):
    def __init__(self, debug_data, title="Text Check Debug", parent=None):
        super().__init__(parent)

        self.debug_data = debug_data or {}
        self.setWindowTitle(title)
        self.resize(1250, 780)

        main_layout = QtWidgets.QVBoxLayout(self)

        result = self.debug_data.get("result", "-")
        missing_pixels = int(self.debug_data.get("missing_pixels", 0))
        missing_ratio = float(self.debug_data.get("missing_ratio", 0.0))
        error = self.debug_data.get("error")

        info = (
            f"Result={result} | Tol={TOLERANCE_PX:.2f}px"
            f" | MinMissing={MIN_MISSING_LENGTH}px"
            f" | MissingPixels={missing_pixels}"
            f" | MissingRatio={missing_ratio:.5f}"
        )
        if error:
            info += f" | {error}"

        lbl_info = QtWidgets.QLabel(info)
        lbl_info.setWordWrap(True)
        main_layout.addWidget(lbl_info)

        grid = QtWidgets.QGridLayout()
        main_layout.addLayout(grid, 1)

        master_binary = self.debug_data.get("master_binary")
        master_skeleton = self.debug_data.get("master_skeleton")
        test_binary = self.debug_data.get("test_binary")
        overlay = self.debug_data.get("overlay")
        distance = self.debug_data.get("distance")
        missing = self.debug_data.get("missing")

        distance_show = self.make_distance_map(distance)

        items = [
            ("1. Master Binary", master_binary),
            ("2. Master Skeleton", master_skeleton),
            ("3. Test Binary", test_binary),
            ("4. Master Skeleton Overlay Test", overlay),
            ("5. Distance Map", distance_show),
            ("6. Missing Mask (Filtered)", missing),
        ]

        self.image_labels = []

        for index, (name, image) in enumerate(items):
            box = QtWidgets.QGroupBox(name)
            box_layout = QtWidgets.QVBoxLayout(box)

            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumSize(320, 250)
            label.setStyleSheet(
                "QLabel{background:#202020;border:1px solid #555;}"
            )

            if image is not None and image.size > 0:
                pixmap = self.image_to_pixmap(image)
                label.setPixmap(
                    pixmap.scaled(
                        380,
                        300,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.FastTransformation
                    )
                )
            else:
                label.setText("No image")
                label.setStyleSheet(
                    "QLabel{background:#202020;color:white;border:1px solid #555;}"
                )

            box_layout.addWidget(label)
            self.image_labels.append(label)

            row = index // 3
            col = index % 3
            grid.addWidget(box, row, col)

    @staticmethod
    def make_distance_map(distance):
        if distance is None or distance.size == 0:
            return None

        # Clip để vùng quanh tolerance dễ nhìn hơn thay vì để background xa
        # chiếm toàn bộ dynamic range.
        max_show = max(5.0, float(TOLERANCE_PX) * 4.0)
        clipped = np.clip(distance, 0.0, max_show)

        distance_u8 = np.uint8(
            np.round(clipped / max_show * 255.0)
        )

        return cv2.applyColorMap(
            distance_u8,
            cv2.COLORMAP_JET
        )

    @staticmethod
    def image_to_pixmap(image):
        image = np.ascontiguousarray(image)

        if image.ndim == 2:
            h, w = image.shape
            qimage = QtGui.QImage(
                image.data,
                w,
                h,
                image.strides[0],
                QtGui.QImage.Format_Grayscale8
            ).copy()
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            h, w, _ = rgb.shape
            qimage = QtGui.QImage(
                rgb.data,
                w,
                h,
                rgb.strides[0],
                QtGui.QImage.Format_RGB888
            ).copy()

        return QtGui.QPixmap.fromImage(qimage)


# ============================================================
# TEST WINDOW
# ============================================================
class TestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deskew + Object ROI + Text Missing Check")
        self.resize(1250, 780)

        self.current_image_path = None
        self.original_image = None
        self.image = None

        self.before_angle = None
        self.final_angle = None
        self.rotation_correction = 0.0

        self.regions = []

        self.master_image = None
        self.master_binary = None
        self.master_skeleton = None
        self.master_path = None

        self.debug_window = None

        self.detector = ROIObjectDetector()
        self.text_checker = TextDefectChecker(
            TOLERANCE_PX,
            MIN_MISSING_LENGTH
        )
        self.image_view = ImageCanvas()

        self.btn_open = QtWidgets.QPushButton("Open Image")
        self.btn_cut_master = QtWidgets.QPushButton("Cut Master")
        self.btn_load_master = QtWidgets.QPushButton("Load Master")
        self.btn_save = QtWidgets.QPushButton("Save Config")
        self.btn_debug = QtWidgets.QPushButton("Show Check Debug")
        self.btn_clear = QtWidgets.QPushButton("Clear ROIs")
        self.lbl_status = QtWidgets.QLabel(
            "Open master image -> draw ROI(s) -> Save Config"
        )

        self.btn_open.clicked.connect(self.open_image)
        self.btn_cut_master.clicked.connect(self.cut_master)
        self.btn_load_master.clicked.connect(self.load_master)
        self.btn_save.clicked.connect(self.save_config)
        self.btn_debug.clicked.connect(self.show_check_debug)
        self.btn_clear.clicked.connect(self.clear_rois)
        self.image_view.roiFinished.connect(self.detect_object)

        self.shortcut_escape = QtWidgets.QShortcut(
            QtGui.QKeySequence("Esc"), self
        )
        self.shortcut_escape.activated.connect(
            self.image_view.cancel_draw
        )

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_cut_master)
        top.addWidget(self.btn_load_master)
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_debug)
        top.addWidget(self.btn_clear)
        top.addWidget(self.lbl_status, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.image_view, 1)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )

        if not path:
            return

        self.current_image_path = Path(path)

        try:
            self.original_image = read_image(self.current_image_path)
        except FileNotFoundError:
            self.lbl_status.setText("Cannot load image")
            return

        try:
            rotated, before, after, correction = deskew_to_zero(
                self.original_image
            )

            self.image = rotated
            self.before_angle = float(before["angle"])
            self.final_angle = float(after["angle"])
            self.rotation_correction = float(correction)

            # Chỉ dùng kết quả deskew.
            # Không tạo / hiển thị bbox label để làm offset nữa.
            self.regions = []
            self.image_view.set_image(self.image)

            if config_exists():
                self.load_config_to_current_image()
            else:
                self.lbl_status.setText(
                    f"Deskew OK | Final={self.final_angle:+.4f} deg"
                    " | No config -> draw ROI(s), then Save Config"
                )

        except Exception as error:
            self.image = self.original_image.copy()
            self.regions = []
            self.image_view.set_image(self.image)
            self.lbl_status.setText(
                f"Deskew failed: {error}"
            )

    def set_master(self, image, path=None):
        if image is None or image.size == 0:
            self.master_image = None
            self.master_binary = None
            self.master_skeleton = None
            self.master_path = None
            return False

        self.master_image = image.copy()
        self.master_skeleton = self.text_checker.create_master_skeleton(
            self.master_image
        )
        self.master_binary = (
            self.text_checker.last_master_binary.copy()
            if self.text_checker.last_master_binary is not None
            else None
        )
        self.master_path = Path(path) if path else None
        return True

    def cut_master(self):
        if self.image is None:
            self.lbl_status.setText("No image.")
            return

        if not self.regions:
            self.lbl_status.setText(
                "Draw ROI first, then click Cut Master."
            )
            return

        # Lấy bbox object nét đứt của ROI vừa tạo gần nhất.
        region = self.regions[-1]
        object_rect = region.get("object_rect")
        if object_rect is None:
            self.lbl_status.setText("No object-fit bbox.")
            return

        x, y, w, h = map(int, object_rect)
        image_h, image_w = self.image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_w, x + w)
        y2 = min(image_h, y + h)

        crop = self.image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            self.lbl_status.setText("Cannot cut Master.")
            return

        default_name = f"{region['name']}_master.png"

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Master",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"
        )

        if not save_path:
            return

        suffix = Path(save_path).suffix.lower()
        ext = suffix if suffix in {".png", ".jpg", ".jpeg"} else ".png"

        if not Path(save_path).suffix:
            save_path += ext

        ok, encoded = cv2.imencode(ext, crop)
        if not ok:
            self.lbl_status.setText("Cannot encode Master.")
            return

        try:
            encoded.tofile(save_path)
        except OSError:
            self.lbl_status.setText("Cannot save Master.")
            return

        self.set_master(crop, save_path)

        self.lbl_status.setText(
            f"Master cut: {Path(save_path).name}"
            f" | Size={crop.shape[1]}x{crop.shape[0]}"
        )

    def load_master(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Master",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not path:
            return

        try:
            master = read_image(Path(path))
        except FileNotFoundError:
            self.lbl_status.setText("Cannot load Master.")
            return

        if not self.set_master(master, path):
            self.lbl_status.setText("Invalid Master.")
            return

        checked = 0

        for region in self.regions:
            if self.check_region_with_loaded_master(region):
                checked += 1

        self.image_view.set_regions(self.regions)

        self.lbl_status.setText(
            f"Master loaded: {Path(path).name}"
            f" | Size={master.shape[1]}x{master.shape[0]}"
            f" | Checked ROI={checked}"
        )

    @staticmethod
    def build_ng_contours(missing_mask, offset_x, offset_y):
        if missing_mask is None or missing_mask.size == 0:
            return []

        # missing_mask là kết quả NG đã được thuật toán cũ lọc.
        # Chỉ nới 1 pixel cho đường viền dễ nhìn trên GUI.
        show_mask = cv2.dilate(
            missing_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1
        )

        contours, _ = cv2.findContours(
            show_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = []

        for contour in contours:
            if contour is None or len(contour) < 2:
                continue

            points = contour.reshape(-1, 2)

            points[:, 0] += int(offset_x)
            points[:, 1] += int(offset_y)

            result.append([
                (int(px), int(py))
                for px, py in points
            ])

        return result

    def check_region_with_loaded_master(self, region):
        if (
            self.image is None
            or self.master_skeleton is None
            or region is None
        ):
            return False

        object_rect = region.get("object_rect")
        if object_rect is None:
            return False

        x, y, w, h = map(int, object_rect)
        image_h, image_w = self.image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_w, x + w)
        y2 = min(image_h, y + h)

        test_crop = self.image[y1:y2, x1:x2]

        check = self.text_checker.check(
            test_crop,
            self.master_skeleton,
            self.master_binary
        )

        # Lưu toàn bộ 6 ảnh debug ngay trong region hiện tại.
        region["debug"] = check

        defects = []

        for defect in check["defects"]:
            dx, dy, dw, dh = defect["bbox"]
            defects.append({
                "length": int(defect["length"]),
                "bbox": (
                    x1 + int(dx),
                    y1 + int(dy),
                    int(dw),
                    int(dh)
                )
            })

        text_mask = None
        test_binary = check.get("test_binary")

        if test_binary is not None and test_binary.size > 0:
            text_mask = {
                "x": x1,
                "y": y1,
                "mask": test_binary.copy()
            }

        ng_contours = self.build_ng_contours(
            check.get("missing"),
            x1,
            y1
        )

        region["result"] = check["result"]
        region["defects"] = defects
        region["ng_contours"] = ng_contours
        region["text_mask"] = text_mask
        region["missing_pixels"] = check["missing_pixels"]
        region["missing_ratio"] = check["missing_ratio"]
        region["error"] = check["error"]

        return True

    def show_check_debug(self):
        available = [
            region
            for region in self.regions
            if region.get("debug") is not None
        ]

        if not available:
            self.lbl_status.setText(
                "Chưa có kết quả check để hiển thị debug."
            )
            return

        if len(available) == 1:
            region = available[0]
        else:
            names = [str(region.get("name", "text")) for region in available]
            selected, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Select ROI Debug",
                "ROI:",
                names,
                0,
                False
            )

            if not ok:
                return

            region = available[names.index(selected)]

        debug = region.get("debug")
        name = str(region.get("name", "text"))

        self.debug_window = DebugImageWindow(
            debug,
            title=f"Text Check Debug - {name}",
            parent=self
        )
        self.debug_window.show()

    def detect_object(self, roi):
        if self.image is None:
            return

        roi = tuple(int(v) for v in roi)
        object_rect = self.detector.detect(self.image, roi)

        if object_rect is None:
            self.lbl_status.setText("No object detected inside ROI.")
            return

        name = self.next_roi_name()

        region = {
            "name": name,

            # rect = bbox do người dùng vẽ, giữ nguyên.
            "rect": roi,

            # object_rect = bbox nét đứt tự fit đúng object bên trong ROI.
            "object_rect": tuple(int(v) for v in object_rect)
        }

        self.regions.append(region)

        if self.master_skeleton is not None:
            self.check_region_with_loaded_master(region)

        self.image_view.set_regions(self.regions)

        rx, ry, rw, rh = roi
        ox, oy, ow, oh = object_rect
        result_text = (
            f" | Result={region.get('result')}"
            if region.get("result")
            else ""
        )

        self.lbl_status.setText(
            f"Added {name}"
            f" | ROI=({rx},{ry},{rw},{rh})"
            f" | Object fit=({ox},{oy},{ow},{oh})"
            f"{result_text}"
        )

    def next_roi_name(self):
        used = {region["name"] for region in self.regions}
        index = 1
        while f"text_{index}" in used:
            index += 1
        return f"text_{index}"

    def clear_rois(self):
        self.regions = []
        self.image_view.set_regions(self.regions)
        self.lbl_status.setText(
            "ROIs cleared in current view. Config is NOT changed until Save Config."
        )

    def save_config(self):
        if self.image is None:
            self.lbl_status.setText("No image.")
            return

        if not self.regions:
            self.lbl_status.setText("No ROI to save.")
            return

        config_rois = []
        h_img, w_img = self.image.shape[:2]

        for region in self.regions:
            roi_x, roi_y, roi_w, roi_h = region["rect"]
            object_rect = region.get("object_rect", region["rect"])
            obj_x, obj_y, obj_w, obj_h = object_rect

            # Crop ROI người dùng vẽ để dùng làm template offset.
            rx1 = max(0, roi_x)
            ry1 = max(0, roi_y)
            rx2 = min(w_img, roi_x + roi_w)
            ry2 = min(h_img, roi_y + roi_h)
            roi_crop = self.image[ry1:ry2, rx1:rx2]

            # Crop đúng bbox object nét đứt để tạo skeleton Master.
            ox1 = max(0, obj_x)
            oy1 = max(0, obj_y)
            ox2 = min(w_img, obj_x + obj_w)
            oy2 = min(h_img, obj_y + obj_h)
            object_crop = self.image[oy1:oy2, ox1:ox2]

            if roi_crop.size == 0 or object_crop.size == 0:
                continue

            master_skeleton = self.text_checker.create_master_skeleton(
                object_crop
            )
            master_binary = (
                self.text_checker.last_master_binary.copy()
                if self.text_checker.last_master_binary is not None
                else self.text_checker.preprocess(object_crop)
            )

            config_rois.append({
                "name": region["name"],
                "type": "text",

                # ROI do người dùng vẽ.
                "x": int(roi_x),
                "y": int(roi_y),
                "w": int(roi_w),
                "h": int(roi_h),

                # Bbox object relative theo ROI.
                "object_x": int(obj_x - roi_x),
                "object_y": int(obj_y - roi_y),
                "object_w": int(obj_w),
                "object_h": int(obj_h),

                # ROI template dùng tìm offset trên ảnh test.
                "template_png": encode_template_png(roi_crop),

                # Binary + Skeleton chỉ lấy đúng object bbox.
                "master_binary_png": encode_gray_png(master_binary),
                "master_skeleton_png": encode_gray_png(master_skeleton)
            })

        data = {
            "version": 5,
            "master": {
                "image_path": (
                    str(self.current_image_path)
                    if self.current_image_path
                    else ""
                ),
                "image_width": int(w_img),
                "image_height": int(h_img)
            },
            "rois": config_rois
        }

        saved_path = save_config_file(data)

        self.lbl_status.setText(
            f"Saved: {saved_path}"
            f" | ROIs={len(config_rois)}"
            f" | Object fit=no padding"
            f" | Tol={TOLERANCE_PX}px"
            f" | Min missing={MIN_MISSING_LENGTH}px"
        )

    def load_config_to_current_image(self):
        if self.image is None:
            return False

        data = load_config()

        if not data:
            return False

        config_rois = data.get("rois", [])

        if not config_rois:
            return False

        # ----------------------------------------------------
        # Tìm offset từ từng ROI master.
        # Mỗi ROI chỉ match với template của chính nó.
        # Sau đó lấy median dx/dy để tránh 1 ROI match sai.
        # ----------------------------------------------------
        matches = []

        for roi in config_rois:
            match = find_roi_offset(
                self.image,
                roi
            )

            if (
                match is not None
                and match["score"] >= OFFSET_MATCH_MIN_SCORE
            ):
                matches.append({
                    "name": str(roi.get("name", "text")),
                    "dx": match["dx"],
                    "dy": match["dy"],
                    "score": match["score"]
                })

        if matches:
            dx = int(round(np.median(
                [item["dx"] for item in matches]
            )))
            dy = int(round(np.median(
                [item["dy"] for item in matches]
            )))

            avg_score = float(np.mean(
                [item["score"] for item in matches]
            ))
        else:
            # Không tìm được offset đáng tin cậy:
            # giữ tọa độ master thay vì detect lại word.
            dx = 0
            dy = 0
            avg_score = 0.0

        mapped_regions = []
        good_count = 0
        ng_count = 0
        total_defects = 0

        image_h, image_w = self.image.shape[:2]

        for roi in config_rois:
            master_roi_x = int(roi.get("x", 0))
            master_roi_y = int(roi.get("y", 0))
            roi_w = int(roi.get("w", 0))
            roi_h = int(roi.get("h", 0))

            roi_x = master_roi_x + dx
            roi_y = master_roi_y + dy

            # Config mới: object bbox được lưu relative theo ROI.
            # Config cũ: nếu chưa có object_x/y/w/h thì dùng toàn ROI.
            object_rel_x = int(roi.get("object_x", 0))
            object_rel_y = int(roi.get("object_y", 0))
            object_w = int(roi.get("object_w", roi_w))
            object_h = int(roi.get("object_h", roi_h))

            object_x = roi_x + object_rel_x
            object_y = roi_y + object_rel_y

            ox1 = max(0, object_x)
            oy1 = max(0, object_y)
            ox2 = min(image_w, object_x + object_w)
            oy2 = min(image_h, object_y + object_h)

            test_crop = self.image[oy1:oy2, ox1:ox2]

            # ------------------------------------------------
            # MASTER BINARY + SKELETON
            # ------------------------------------------------
            # Nếu người dùng Load Master thủ công thì ưu tiên Master đang load.
            master_binary = (
                self.master_binary.copy()
                if self.master_binary is not None
                else decode_gray_png(
                    roi.get("master_binary_png", "")
                )
            )

            master_skeleton = (
                self.master_skeleton.copy()
                if self.master_skeleton is not None
                else decode_gray_png(
                    roi.get("master_skeleton_png", "")
                )
            )

            # Hỗ trợ config cũ chưa có master_binary_png / skeleton.
            # Dùng template ROI rồi crop đúng object bbox relative để dựng lại.
            if master_binary is None or master_skeleton is None:
                template_gray = decode_template_png(
                    roi.get("template_png", "")
                )

                if template_gray is not None and template_gray.size > 0:
                    th, tw = template_gray.shape[:2]
                    rel_x1 = max(0, object_rel_x)
                    rel_y1 = max(0, object_rel_y)
                    rel_x2 = min(tw, object_rel_x + object_w)
                    rel_y2 = min(th, object_rel_y + object_h)

                    object_template = template_gray[
                        rel_y1:rel_y2,
                        rel_x1:rel_x2
                    ]

                    if object_template.size > 0:
                        object_template_bgr = cv2.cvtColor(
                            object_template,
                            cv2.COLOR_GRAY2BGR
                        )

                        if master_binary is None:
                            master_binary = self.text_checker.preprocess(
                                object_template_bgr
                            )

                        if master_skeleton is None:
                            master_skeleton = skeletonize(
                                master_binary > 0
                            ).astype(np.uint8) * 255

            check = self.text_checker.check(
                test_crop,
                master_skeleton,
                master_binary
            )

            text_mask = None
            test_binary = check.get("test_binary")

            if test_binary is not None and test_binary.size > 0:
                text_mask = {
                    "x": ox1,
                    "y": oy1,
                    "mask": test_binary.copy()
                }

            absolute_defects = []
            for defect in check["defects"]:
                fx, fy, fw, fh = defect["bbox"]
                absolute_defects.append({
                    "length": int(defect["length"]),
                    "bbox": (
                        ox1 + int(fx),
                        oy1 + int(fy),
                        int(fw),
                        int(fh)
                    )
                })

            ng_contours = self.build_ng_contours(
                check.get("missing"),
                ox1,
                oy1
            )

            result = check["result"]

            if result == "GOOD":
                good_count += 1
            else:
                ng_count += 1

            total_defects += len(absolute_defects)

            mapped_regions.append({
                "name": str(roi.get("name", "text")),

                # ROI người dùng vẽ.
                "rect": (
                    roi_x,
                    roi_y,
                    roi_w,
                    roi_h
                ),

                # Bbox object nét đứt.
                "object_rect": (
                    object_x,
                    object_y,
                    object_w,
                    object_h
                ),

                "result": result,
                "defects": absolute_defects,
                "ng_contours": ng_contours,
                "text_mask": text_mask,
                "missing_pixels": check["missing_pixels"],
                "missing_ratio": check["missing_ratio"],
                "error": check["error"],
                "debug": check
            })

        self.regions = mapped_regions
        self.image_view.set_regions(self.regions)

        offset_text = (
            f"Offset dx={dx:+d}, dy={dy:+d}"
            f" | Matched ROI={len(matches)}/{len(config_rois)}"
            f" | Avg score={avg_score:.3f}"
            if matches
            else "Offset match failed | dx=0, dy=0"
        )

        self.lbl_status.setText(
            f"{offset_text}"
            f" | GOOD={good_count}"
            f" | NG={ng_count}"
            f" | Defects={total_defects}"
        )

        return True



def main():
    app = QtWidgets.QApplication(sys.argv)

    window = TestWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()