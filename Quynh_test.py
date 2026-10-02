import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIG =================

TEST_FOLDER = Path(r"D:\check_textLabel\img_label")
SAVE_FOLDER = Path(r"D:\check_textLabel\save_folder")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

COLOR_DELTA_Y = 2.2
COLOR_DELTA_X = 2.2

SMOOTH_KERNEL = 31
MIN_GAP_SIZE = 8
# Barcode/text only dips ~2.3 below label color; real gutters dip ~4–7.
MIN_GAP_DEPTH = 3.2
MIN_GAP_DEPTH_RELAX = 2.5
# Column profile: high percentile follows label paper, not dense FCC/barcode ink.
COL_PERCENTILE = 80
LABEL_HEIGHT_HINT = 1260

SEARCH_IN = 30
SEARCH_OUT = 40
SAMPLE_STEP = 5

ANGLE_TOLERANCE = 0.05
MAX_ROTATE_ITER = 2

SHOW_RESULT = True

SPEC = {
    "left_min": 10,
    "right_min": 10,
    "top_min": 10,
    "bottom_min": 10,
}


# ================= IO =================


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".png"
        path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    buf.tofile(str(path))
    return path


# ================= SMOOTH =================


def smooth_profile(profile, kernel=31):
    if kernel % 2 == 0:
        kernel += 1
    profile = np.asarray(profile, dtype=np.float32)
    return cv2.GaussianBlur(profile.reshape(-1, 1), (1, kernel), 0).ravel()


# ================= FIND SEGMENTS =================


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


def strong_gaps(profile, gaps, center, min_depth=MIN_GAP_DEPTH):
    """Drop shallow valleys (barcode/text), keep real label gutters."""
    keep = []
    for gs, ge in gaps:
        depth = float(center) - float(profile[gs : ge + 1].min())
        if depth >= min_depth:
            keep.append((gs, ge))
    return keep


def pick_side_gaps(gaps, mid, side):
    if side == "left":
        return [g for g in gaps if g[1] < mid]
    return [g for g in gaps if g[0] > mid]


def gaps_at_depth(profile, center, delta, mid, depth):
    thr = center - delta
    gaps = find_segments(profile < thr, MIN_GAP_SIZE)
    gaps = strong_gaps(profile, gaps, center, depth)
    return pick_side_gaps(gaps, mid, "left"), pick_side_gaps(gaps, mid, "right")


def snap_outer_edge(profile, approx, paper, delta, from_low):
    """First paper pixel walking in from the table — not the blurred inner gap."""
    profile = np.asarray(profile, dtype=np.float32)
    n = len(profile)
    approx = int(np.clip(approx, 0, n - 1))
    thr = float(paper) - float(delta)
    if from_low:
        start = max(0, approx - 50)
        end = min(n, approx + 6)
        for i in range(start, end):
            if profile[i] >= thr:
                return i
        return approx
    start = min(n - 1, approx + 50)
    end = max(0, approx - 6)
    for i in range(start, end, -1):
        if profile[i] >= thr:
            return i + 1
    return approx


def detect_label(image):
    H, W = image.shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)

    x1 = int(W * 0.12)
    x2 = int(W * 0.93)
    row = np.median(a[:, x1:x2], axis=1)
    row_fine = smooth_profile(row, 5)
    row = smooth_profile(row, SMOOTH_KERNEL)
    row_center = float(np.median(row[int(H * 0.35) : int(H * 0.65)]))
    row_thr = row_center - COLOR_DELTA_Y
    cy = H // 2

    row_gaps = find_segments(row < row_thr, MIN_GAP_SIZE)
    upper = pick_side_gaps(
        strong_gaps(row, row_gaps, row_center, MIN_GAP_DEPTH), cy, "left"
    )
    lower = pick_side_gaps(
        strong_gaps(row, row_gaps, row_center, MIN_GAP_DEPTH), cy, "right"
    )
    if not upper or not lower:
        relaxed = strong_gaps(row, row_gaps, row_center, MIN_GAP_DEPTH_RELAX)
        if not upper:
            upper = pick_side_gaps(relaxed, cy, "left")
        if not lower:
            lower = pick_side_gaps(relaxed, cy, "right")

    if not upper and not lower:
        raise RuntimeError("Không tìm được TOP/BOTTOM")

    if upper and lower:
        top = max(upper, key=lambda g: g[1])[1] + 1
        bottom = min(lower, key=lambda g: g[0])[0] - 1
    elif lower:
        bottom = min(lower, key=lambda g: g[0])[0] - 1
        top = max(0, bottom - LABEL_HEIGHT_HINT)
    else:
        top = max(upper, key=lambda g: g[1])[1] + 1
        bottom = min(H, top + LABEL_HEIGHT_HINT)

    h = bottom - top
    if h < 200:
        raise RuntimeError("Không tìm được TOP/BOTTOM")

    top = snap_outer_edge(row_fine, top, row_center, COLOR_DELTA_Y, True)
    bottom = snap_outer_edge(row_fine, bottom, row_center, COLOR_DELTA_Y, False)
    h = bottom - top
    if h < 200:
        raise RuntimeError("Không tìm được TOP/BOTTOM")

    y1 = top + int(h * 0.08)
    y2 = bottom - int(h * 0.08)
    strip = a[y1:y2, :]
    col_fine = smooth_profile(np.percentile(strip, COL_PERCENTILE, axis=0), 5)
    col_med = smooth_profile(np.median(strip, axis=0), SMOOTH_KERNEL)
    col_p = smooth_profile(np.percentile(strip, COL_PERCENTILE, axis=0), SMOOTH_KERNEL)
    col_center = float(np.median(col_p[int(W * 0.35) : int(W * 0.65)]))
    col_thr = col_center - COLOR_DELTA_X
    cx = W // 2

    def lr_from(profile, center, depth):
        return gaps_at_depth(profile, center, COLOR_DELTA_X, cx, depth)

    left_gaps, right_gaps = lr_from(col_p, col_center, MIN_GAP_DEPTH)
    if not left_gaps or not right_gaps:
        left_gaps, right_gaps = lr_from(col_p, col_center, MIN_GAP_DEPTH_RELAX)

    med_center = float(np.median(col_med[int(W * 0.35) : int(W * 0.65)]))
    med_left, med_right = lr_from(col_med, med_center, MIN_GAP_DEPTH)
    if not med_left or not med_right:
        med_left, med_right = lr_from(col_med, med_center, MIN_GAP_DEPTH_RELAX)

    if not left_gaps or not right_gaps:
        left_gaps, right_gaps = med_left, med_right
    if not left_gaps or not right_gaps:
        raise RuntimeError("Không tìm được LEFT/RIGHT")

    left = max(left_gaps, key=lambda g: g[1])[1] + 1
    right = min(right_gaps, key=lambda g: g[0])[0] - 1
    if med_left:
        left = min(left, max(med_left, key=lambda g: g[1])[1] + 1)
    if med_right:
        right = max(right, min(med_right, key=lambda g: g[0])[0] - 1)

    left = snap_outer_edge(col_fine, left, col_center, COLOR_DELTA_X, True)
    right = snap_outer_edge(col_fine, right, col_center, COLOR_DELTA_X, False)

    left = max(0, min(left, W - 1))
    right = max(left + 1, min(right, W))
    top = max(0, min(top, H - 1))
    bottom = max(top + 1, min(bottom, H))

    debug = {
        "row": row,
        "row_center": row_center,
        "row_thr": row_thr,
        "row_gaps": row_gaps,
        "col": col_p,
        "col_center": col_center,
        "col_thr": col_thr,
        "col_gaps": find_segments(col_p < col_thr, MIN_GAP_SIZE),
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
    }
    return left, right, top, bottom, debug


def crop_label(image, left, right, top, bottom):
    """Crop the detected label region from the full image."""
    return image[top:bottom, left:right]


# ================= ANGLE + FULL-IMAGE ROTATE =================


def crossing_from_outside(profile, expected_index, outside_low):
    """Index where paper starts, walking in from the table side."""
    profile = np.asarray(profile, dtype=np.float32)
    n = len(profile)
    if n < 8:
        return None, 0.0
    expected_index = int(np.clip(round(expected_index), 2, n - 3))
    if outside_low:
        n_out = max(3, expected_index)
        out_ref = float(np.median(profile[: n_out // 2 + 1]))
        in_ref = float(np.median(profile[min(n - 3, expected_index + 1) :]))
    else:
        n_out = max(3, n - expected_index)
        out_ref = float(np.median(profile[-(n_out // 2 + 1) :]))
        in_ref = float(np.median(profile[: max(3, expected_index)]))
    contrast = abs(in_ref - out_ref)
    if contrast < 0.8:
        return None, contrast
    mid = 0.5 * (in_ref + out_ref)
    paper_high = in_ref > out_ref
    if outside_low:
        indices = range(n)
    else:
        indices = range(n - 1, -1, -1)
    for i in indices:
        hit = profile[i] >= mid if paper_high else profile[i] <= mid
        if hit:
            return i, contrast
    return expected_index, contrast


def strongest_jump(profile_l, profile_a):
    """Index of the strongest paper/table jump (follows a tilted edge)."""
    p_l = np.asarray(profile_l, dtype=np.float32)
    p_a = np.asarray(profile_a, dtype=np.float32)
    if len(p_l) < 6:
        return None
    g_l = np.diff(p_l)
    g_a = np.diff(p_a)
    candidates = [
        (float(np.max(g_l)), int(np.argmax(g_l))),
        (float(np.max(-g_l)), int(np.argmax(-g_l))),
        (float(np.max(g_a)), int(np.argmax(g_a))),
        (float(np.max(-g_a)), int(np.argmax(-g_a))),
    ]
    peak, idx = max(candidates, key=lambda item: item[0])
    if peak < 0.4:
        return None
    return idx


def best_crossing(profile_l, profile_a, expected_index, outside_low):
    """Use L or a — whichever has the stronger paper/table jump."""
    idx_l, con_l = crossing_from_outside(profile_l, expected_index, outside_low)
    idx_a, con_a = crossing_from_outside(profile_a, expected_index, outside_low)
    if idx_l is None and idx_a is None:
        return None
    if idx_a is None or (idx_l is not None and con_l >= con_a):
        return idx_l
    return idx_a


def find_edge_points(image, left, right, top, bottom, search_out=None, search_in=None):
    H, W = image.shape[:2]
    if search_out is None:
        search_out = SEARCH_OUT
    if search_in is None:
        search_in = SEARCH_IN
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ch_l = cv2.GaussianBlur(lab[:, :, 0].astype(np.float32), (5, 5), 0)
    ch_a = cv2.GaussianBlur(lab[:, :, 1].astype(np.float32), (5, 5), 0)

    bbox_w = right - left
    bbox_h = bottom - top
    margin_x = max(10, int(bbox_w * 0.08))
    margin_y = max(10, int(bbox_h * 0.08))

    top_points, bottom_points = [], []
    left_points, right_points = [], []

    y_start = max(1, top - search_out)
    y_end = min(H - 2, top + search_in)
    for x in range(left + margin_x, right - margin_x, SAMPLE_STEP):
        idx = strongest_jump(
            ch_l[y_start : y_end + 1, x],
            ch_a[y_start : y_end + 1, x],
        )
        if idx is not None:
            top_points.append([x, y_start + idx])

    y_start = max(1, bottom - search_in)
    y_end = min(H - 2, bottom + min(search_out, 12))
    expected = bottom - y_start
    for x in range(left + margin_x, right - margin_x, SAMPLE_STEP):
        idx = best_crossing(
            ch_l[y_start : y_end + 1, x],
            ch_a[y_start : y_end + 1, x],
            expected,
            False,
        )
        if idx is not None:
            bottom_points.append([x, y_start + idx])

    x_start = max(1, left - search_out)
    x_end = min(W - 2, left + search_in)
    expected = left - x_start
    for y in range(top + margin_y, bottom - margin_y, SAMPLE_STEP):
        idx = best_crossing(
            ch_l[y, x_start : x_end + 1],
            ch_a[y, x_start : x_end + 1],
            expected,
            True,
        )
        if idx is not None:
            left_points.append([x_start + idx, y])

    x_start = max(1, right - search_in)
    x_end = min(W - 2, right + search_out)
    expected = right - x_start
    for y in range(top + margin_y, bottom - margin_y, SAMPLE_STEP):
        idx = best_crossing(
            ch_l[y, x_start : x_end + 1],
            ch_a[y, x_start : x_end + 1],
            expected,
            False,
        )
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


def robust_inliers(values):
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    threshold = max(1.5, 3.0 * mad)
    good = values[np.abs(values - median) <= threshold]
    if len(good) == 0:
        return values
    return good


def get_tight_bbox(top_points, bottom_points, left_points, right_points, image_shape):
    """AABB covering paper-edge samples so residual tilt does not clip corners."""
    H, W = image_shape[:2]
    top = int(np.floor(robust_inliers(top_points[:, 1]).min()))
    bottom = int(np.ceil(robust_inliers(bottom_points[:, 1]).max())) + 1
    left = int(np.floor(robust_inliers(left_points[:, 0]).min()))
    right = int(np.ceil(robust_inliers(right_points[:, 0]).max())) + 1
    left = max(0, left)
    top = max(0, top)
    right = min(W, right)
    bottom = min(H, bottom)
    if right - left < 50 or bottom - top < 50:
        raise RuntimeError("Tight bbox quá nhỏ")
    return left, right, top, bottom


def cover_paper_bbox(image, left, right, top, bottom):
    """Expand 1D box outward to paper corners; never shrink into the label."""
    left0, right0, top0, bottom0 = left, right, top, bottom
    try:
        top_pts, bot_pts, left_pts, right_pts = find_edge_points(
            image, left, right, top, bottom
        )
        el, er, et, eb = get_tight_bbox(
            top_pts, bot_pts, left_pts, right_pts, image.shape
        )
    except RuntimeError:
        return left0, right0, top0, bottom0
    max_out = 24
    top = min(top0, max(et, top0 - max_out))
    bottom = max(bottom0, min(eb, bottom0 + max_out))
    left = min(left0, max(el, left0 - max_out))
    right = max(right0, min(er, right0 + max_out))
    H, W = image.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(W, right)
    bottom = min(H, bottom)
    return left, right, top, bottom


def refine_tight_bbox(image, left, right, top, bottom):
    """Keep the full paper after rotate — do not peel into the label."""
    return cover_paper_bbox(image, left, right, top, bottom)


def robust_fit_line(points, iterations=3):
    if len(points) < 10:
        raise RuntimeError("Không đủ điểm fit line")
    current = points.copy()
    for _ in range(iterations):
        line = cv2.fitLine(current, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(line[0][0]), float(line[1][0])
        x0, y0 = float(line[2][0]), float(line[3][0])
        distance = np.abs((current[:, 0] - x0) * vy - (current[:, 1] - y0) * vx)
        distance /= np.hypot(vx, vy)
        median = np.median(distance)
        mad = np.median(np.abs(distance - median))
        threshold = max(1.5, median + 3.0 * mad)
        keep = distance <= threshold
        if np.count_nonzero(keep) < 10:
            break
        current = current[keep]
    line = cv2.fitLine(current, cv2.DIST_L2, 0, 0.01, 0.01)
    return (
        float(line[0][0]),
        float(line[1][0]),
        float(line[2][0]),
        float(line[3][0]),
        current,
    )


def get_edge_angle(points):
    vx, vy, x0, y0, inliers = robust_fit_line(points)
    dist = np.abs((inliers[:, 0] - x0) * vy - (inliers[:, 1] - y0) * vx)
    dist /= max(np.hypot(vx, vy), 1e-6)
    rms = float(np.sqrt(np.mean(dist**2)))
    xs, ys = inliers[:, 0], inliers[:, 1]
    x_lo = np.percentile(xs, 20)
    x_hi = np.percentile(xs, 80)
    left = inliers[xs <= x_lo]
    right = inliers[xs >= x_hi]
    if len(left) >= 5 and len(right) >= 5:
        x1, y1 = float(np.median(left[:, 0])), float(np.median(left[:, 1]))
        x2, y2 = float(np.median(right[:, 0])), float(np.median(right[:, 1]))
        angle = -np.degrees(np.arctan2(y2 - y1, x2 - x1))
    else:
        angle = -np.degrees(np.arctan2(vy, vx))
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return float(angle), inliers, rms


def get_label_angle(top_points, bottom_points):
    """Skew from TOP + BOTTOM long edges. Average when they agree."""
    angle_top, top_inliers, rms_top = get_edge_angle(top_points)
    angle_bottom, bottom_inliers, rms_bottom = get_edge_angle(bottom_points)
    if abs(angle_top - angle_bottom) > 0.2:
        angle = angle_top
        print(
            f"    NOTE: TOP {angle_top:+.3f} deg (rms {rms_top:.2f}) vs "
            f"BOTTOM {angle_bottom:+.3f} deg (rms {rms_bottom:.2f}) "
            f"-> dung TOP {angle:+.3f} deg"
        )
    else:
        angle = 0.5 * (angle_top + angle_bottom)
    return angle, angle_top, angle_bottom, top_inliers, bottom_inliers


def measure_label_angle(image, left, right, top, bottom):
    """Angle of the label from TOP and BOTTOM paper edges only."""
    top_pts, bottom_pts, _lp, _rp = find_edge_points(image, left, right, top, bottom)
    angle, angle_top, angle_bottom, _, _ = get_label_angle(top_pts, bottom_pts)
    return angle, angle_top, angle_bottom, top_pts, bottom_pts


def draw_edge_line(image, points, color):
    """Draw the fitted TOP or BOTTOM edge used for rotation."""
    if points is None or len(points) < 10:
        return
    vx, vy, x0, y0, _ = robust_fit_line(points)
    H, W = image.shape[:2]
    if abs(vx) < 1e-6:
        x1, y1, x2, y2 = int(x0), 0, int(x0), H - 1
    else:
        y1 = int(y0 + (0 - x0) * vy / vx)
        y2 = int(y0 + (W - 1 - x0) * vy / vx)
        x1, x2 = 0, W - 1
    cv2.line(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)


def _shift_points(points, dx, dy):
    if points is None or len(points) == 0:
        return points
    shifted = np.asarray(points, dtype=np.float32).copy()
    shifted[:, 0] += dx
    shifted[:, 1] += dy
    return shifted


def rotate_image(image, rotation_angle):
    """warpAffine the whole frame; extra black canvas so the label is not clipped."""
    H, W = image.shape[:2]
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    extra = 80
    new_W = int(np.ceil(H * sin + W * cos)) + extra
    new_H = int(np.ceil(H * cos + W * sin)) + extra
    M[0, 2] += new_W / 2.0 - center[0]
    M[1, 2] += new_H / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        M,
        (new_W, new_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def deskew_full_image(image):
    """Rotate the whole image using TOP+BOTTOM edges until they are horizontal."""
    left, right, top, bottom, _debug = detect_label(image)
    angle_before, angle_top, angle_bottom, top_pts, bottom_pts = measure_label_angle(
        image, left, right, top, bottom
    )

    total_correction = 0.0
    current = image
    residual = angle_before

    for iteration in range(MAX_ROTATE_ITER):
        left, right, top, bottom, debug = detect_label(current)
        residual, angle_top, angle_bottom, top_pts, bottom_pts = measure_label_angle(
            current, left, right, top, bottom
        )
        print(
            f"    ITER {iteration + 1} | "
            f"top={angle_top:+.4f} deg bottom={angle_bottom:+.4f} deg "
            f"xoay={residual:+.4f} deg"
        )
        if abs(residual) <= ANGLE_TOLERANCE:
            break
        total_correction += -residual
        current = rotate_image(image, total_correction)

    left, right, top, bottom, debug = detect_label(current)
    try:
        left, right, top, bottom = refine_tight_bbox(current, left, right, top, bottom)
    except RuntimeError:
        pass
    residual, angle_top, angle_bottom, top_pts, bottom_pts = measure_label_angle(
        current, left, right, top, bottom
    )
    info = {
        "angle_before": float(angle_before),
        "rotation": float(total_correction),
        "angle_after": float(residual),
        "angle_top": float(angle_top),
        "angle_bottom": float(angle_bottom),
        "top_pts": top_pts,
        "bottom_pts": bottom_pts,
    }
    return current, left, right, top, bottom, debug, info


# ================= TEXT + MARGINS =================


def find_text_bbox(crop):
    """Find the text/barcode region inside the already-cropped label."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid:
        return None

    bx, by, bw_, bh_ = cv2.boundingRect(np.vstack(valid))
    inner = bw[by : by + bh_, bx : bx + bw_]
    coords = cv2.findNonZero(inner)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        bx += rx
        by += ry
        bw_ = rw
        bh_ = rh
    return bx, by, bw_, bh_


def measure_margins(crop, text_box):
    """Distance from text bbox to the 4 edges of the cropped label."""
    if text_box is None:
        return None
    h, w = crop.shape[:2]
    bx, by, bw_, bh_ = text_box
    dl = bx
    dr = w - (bx + bw_)
    dt = by
    db = h - (by + bh_)
    ok = (
        dl >= SPEC["left_min"]
        and dr >= SPEC["right_min"]
        and dt >= SPEC["top_min"]
        and db >= SPEC["bottom_min"]
    )
    return {
        "L": dl,
        "R": dr,
        "T": dt,
        "B": db,
        "ok": ok,
        "box": text_box,
    }


def draw_text_margins(crop, margins):
    """Draw outer label, inner text box, and L/R/T/B distances."""
    vis = crop.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (1, 1), (w - 2, h - 2), (255, 0, 0), 2)

    if margins is None:
        cv2.putText(
            vis,
            "NO TEXT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return vis

    bx, by, bw_, bh_ = margins["box"]
    dl, dr, dt, db = margins["L"], margins["R"], margins["T"], margins["B"]
    color = (0, 200, 0) if margins["ok"] else (0, 0, 255)
    cv2.rectangle(vis, (bx, by), (bx + bw_ - 1, by + bh_ - 1), color, 2)

    mid_y = by + bh_ // 2
    mid_x = bx + bw_ // 2
    cv2.line(vis, (0, mid_y), (bx, mid_y), color, 2)
    cv2.line(vis, (bx + bw_, mid_y), (w - 1, mid_y), color, 2)
    cv2.line(vis, (mid_x, 0), (mid_x, by), color, 2)
    cv2.line(vis, (mid_x, by + bh_), (mid_x, h - 1), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"L={dl}px", (max(4, bx // 4), mid_y - 8), font, 0.6, color, 2)
    cv2.putText(vis, f"R={dr}px", (bx + bw_ + 4, mid_y - 8), font, 0.6, color, 2)
    cv2.putText(
        vis, f"T={dt}px", (mid_x + 6, max(22, dt // 2 + 10)), font, 0.6, color, 2
    )
    cv2.putText(
        vis,
        f"B={db}px",
        (mid_x + 6, min(h - 8, by + bh_ + db // 2 + 8)),
        font,
        0.6,
        color,
        2,
    )
    cv2.putText(vis, "OK" if margins["ok"] else "NG", (8, 28), font, 0.9, color, 2)
    return vis


def show_detection(
    image,
    left,
    right,
    top,
    bottom,
    crop,
    margins,
    debug,
    title,
    angle_info,
    process_ms,
):
    """Show rotated frame + bbox, cropped label, profiles, angles, and time."""
    crop_vis = draw_text_margins(crop, margins)
    draw_edge_line(
        crop_vis,
        _shift_points(angle_info.get("top_pts"), -left, -top),
        (0, 255, 0),
    )
    draw_edge_line(
        crop_vis,
        _shift_points(angle_info.get("bottom_pts"), -left, -top),
        (0, 255, 255),
    )

    result = image.copy()
    cv2.rectangle(result, (left, top), (right, bottom), (255, 0, 0), 4)
    if margins is not None:
        bx, by, bw_, bh_ = margins["box"]
        color = (0, 200, 0) if margins["ok"] else (0, 0, 255)
        cv2.rectangle(
            result,
            (left + bx, top + by),
            (left + bx + bw_, top + by + bh_),
            color,
            3,
        )

    before = angle_info["angle_before"]
    rotation = angle_info["rotation"]
    after = angle_info["angle_after"]
    ang_top = angle_info["angle_top"]
    ang_bot = angle_info["angle_bottom"]
    draw_edge_line(result, angle_info.get("top_pts"), (0, 255, 0))
    draw_edge_line(result, angle_info.get("bottom_pts"), (0, 255, 255))

    H, W = result.shape[:2]
    font_scale = np.clip(W / 1900.0, 1.2, 2.2)
    thickness = max(2, int(font_scale * 2))
    cv2.putText(
        result,
        f"TOP={ang_top:+.3f} deg  BOTTOM={ang_bot:+.3f} deg  "
        f"XOAY={rotation:+.3f} deg  con lai={after:+.3f} deg",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    status = "NO TEXT" if margins is None else ("OK" if margins["ok"] else "NG")
    margin_txt = ""
    if margins is not None:
        margin_txt = (
            f" | L={margins['L']} R={margins['R']} "
            f"T={margins['T']} B={margins['B']} px"
        )
    fig.suptitle(
        f"{title} | {status}{margin_txt} | {process_ms:.1f} ms | "
        f"TOP {ang_top:+.3f}°  BOTTOM {ang_bot:+.3f}°  "
        f"xoay {rotation:+.3f}°  còn lại {after:+.3f}°",
        fontsize=12,
    )

    axes[0, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(
        f"1) Xoay theo TOP+BOTTOM | "
        f"T {ang_top:+.3f}° B {ang_bot:+.3f}° → xoay {rotation:+.3f}°"
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(crop_vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"2) Crop sát mép giấy | còn lại {after:+.3f}°")
    axes[0, 1].axis("off")

    row = debug["row"]
    ys = np.arange(len(row))
    ax_row = axes[1, 0]
    ax_row.plot(row, ys, color="tab:blue", linewidth=1.2, label="row profile")
    ax_row.axvline(debug["row_thr"], color="tab:red", linestyle="--", label="threshold")
    ax_row.axvline(
        debug["row_center"], color="tab:orange", linestyle=":", label="center"
    )
    for gs, ge in debug["row_gaps"]:
        ax_row.axhspan(gs, ge, color="0.7", alpha=0.35)
    ax_row.axhline(top, color="tab:green", linewidth=1.5, label="top/bottom")
    ax_row.axhline(bottom, color="tab:green", linewidth=1.5)
    ax_row.set_ylim(len(row) - 1, 0)
    ax_row.set_xlabel("LAB a")
    ax_row.set_ylabel("y")
    ax_row.set_title("Xử lý TOP/BOTTOM (khe xám, cạnh xanh)")
    ax_row.legend(loc="upper right", fontsize=8)
    ax_row.grid(True, alpha=0.3)

    col = debug["col"]
    xs = np.arange(len(col))
    ax_col = axes[1, 1]
    ax_col.plot(xs, col, color="tab:blue", linewidth=1.2, label="col profile")
    ax_col.axhline(debug["col_thr"], color="tab:red", linestyle="--", label="threshold")
    ax_col.axhline(
        debug["col_center"], color="tab:orange", linestyle=":", label="center"
    )
    for gs, ge in debug["col_gaps"]:
        ax_col.axvspan(gs, ge, color="0.7", alpha=0.35)
    ax_col.axvline(left, color="tab:green", linewidth=1.5, label="left/right")
    ax_col.axvline(right, color="tab:green", linewidth=1.5)
    ax_col.set_xlabel("x")
    ax_col.set_ylabel("LAB a")
    ax_col.set_title("Xử lý LEFT/RIGHT (khe xám, cạnh xanh)")
    ax_col.legend(loc="upper right", fontsize=8)
    ax_col.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ================= MAIN =================


def main():
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in TEST_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    print(f"Folder: {TEST_FOLDER}")
    print(f"Save:   {SAVE_FOLDER}")
    print(f"Số ảnh: {len(paths)}")
    print("=" * 60)

    for i, path in enumerate(paths, 1):
        try:
            image = read_image(path)
            t0 = time.perf_counter()
            rotated, left, right, top, bottom, debug, angle_info = deskew_full_image(
                image
            )
            crop = crop_label(rotated, left, right, top, bottom)
            text_box = find_text_bbox(crop)
            margins = measure_margins(crop, text_box)
            out_path = save_image(SAVE_FOLDER / path.name, crop)
            process_ms = (time.perf_counter() - t0) * 1000.0

            w = right - left
            h = bottom - top
            print(
                f"[{i}/{len(paths)}] {path.name} | "
                f"{process_ms:.1f} ms | "
                f"TOP={angle_info['angle_top']:+.4f}° "
                f"BOTTOM={angle_info['angle_bottom']:+.4f}° "
                f"xoay={angle_info['rotation']:+.4f}° "
                f"còn lại={angle_info['angle_after']:+.4f}° | "
                f"SIZE={w}x{h} | saved {out_path.name}"
            )
            if margins is None:
                print("    text: NO TEXT")
            else:
                print(
                    f"    text margin L={margins['L']}px R={margins['R']}px "
                    f"T={margins['T']}px B={margins['B']}px | "
                    f"{'OK' if margins['ok'] else 'NG'}"
                )

            if SHOW_RESULT:
                show_detection(
                    rotated,
                    left,
                    right,
                    top,
                    bottom,
                    crop,
                    margins,
                    debug,
                    path.name,
                    angle_info,
                    process_ms,
                )

        except Exception as e:
            print(f"[{i}/{len(paths)}] {path.name} | FAIL: {e}")


if __name__ == "__main__":
    main()
