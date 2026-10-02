import cv2
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import re
import time

sys.path.insert(0, r"D:\Backup\pythonProject_AD220")
import Quynh_test as label_pipe

# ================= CONFIG =================
# Inspect va build spec tach hang bang muc. OCR tat.
USE_OCR = False
OCR_CONF = 0.5
ocr = None


def get_ocr():
    """Load PaddleOCR khi that su can (build spec hoac USE_OCR=True)."""
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang="en", use_angle_cls=True)
    return ocr
TEST_FOLDER = Path(r"D:\dataset\test_Anomalib\test_Anomalib\img_label")
OK_FOLDER = Path(r"D:\dataset\test_Anomalib\test_Anomalib\img_label\OK")
NG_FOLDER = Path(r"D:\dataset\test_Anomalib\test_Anomalib\img_label\NG")
SPEC_FILE = Path(__file__).resolve().parent / "spec.json"
SPEC_TEMPL_DIR = Path(__file__).resolve().parent / "spec_templates" 
# True: build spec neu chua co (hoac spec cu khong co key lines). False: chi inspect.
BUILD_SPEC = False
# True: bat buoc build lai ke ca khi da co spec.json.
FORCE_BUILD_SPEC = False
TEMP_PATH = Path(r"D:\dataset\test_Anomalib\test_Anomalib\tempt.jpg")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ROI trên crop tem đã xoay + detect (x, y, w, h) — This_device 1+2+3
# CHECK_AREA = (2052, 40, 1260, 295)
CHECK_AREA = (2053, 331, 1160, 405)
ORIGIN_MASTER = (2073, 785)
TEMP_RATE = 0.7
MASTER = label_pipe.read_image(TEMP_PATH)
MIN_LINE_H = 10
MIN_WORD_GAP_FRAC = 0.20
PUNCT_WIDTH_FRAC = 0.28
LINE_BOX_PAD = 1
CONTENT_RATIO = 0.82
HEIGHT_MIN_STD = 2.5
WIDTH_MIN_STD = 8.0
# Nguong mac dinh khi build spec tu folder OK. Sua trong spec.json["defaults"]
# hoac spec["lines"][i]["slots"][j]["thresholds"] de nới/thắt từng ô.
DEFAULT_THRESH = {
    "k": 4.0,
    "area_rel": 0.12,
    "area_abs": 50.0,
    "density_rel": 0.12,
    "density_abs": 0.035,
    "stroke_rel": 0.12,
    "stroke_abs": 0.25,
    "laplacian_ratio": 0.60,
    "content_ratio": 0.82,
    "height_rel": 0.08,
    "height_abs": 3.0,
    "width_rel": 0.03,
    "width_abs": 12.0,
    "check_content": False,
    "check_word_count": False,
    "check_chars": False,
    "check_height": True,
    "check_width": True,
    "check_area": False,
    "check_density": False,
    "check_stroke": False,
    "check_laplacian": True,
    "check_template": False,
    "check_ink": True,
    "check_mark": True,
    "check_letter": True,
    "check_fade": True,
    "match_slack": 0.08,
    "match_min": 0.65,
    "ink_slack": 0.06,
    "ink_min": 0.62,
    "fade_slack": 0.06,
    "fade_min": 0.28,
    "fade_rel_slack": 0.03,
    "fade_rel_min": 0.40,
    "fade_dip_slack": 0.04,
    "fade_dip_min": 0.55,
    "cover_slack": 0.12,
    "cover_min": 0.58,
    "mark_slack": 0.12,
    "mark_min": 0.80,
    "letter_slack": 0.22,
    "letter_min": 0.50,
    "skel_slack": 0.05,
    "skel_min": 0.80,
    "check_punct": False,
    "check_dot": True,
    "check_shift": True,
    "check_extra": True,
    "punct_min": 0.45,
    "shift_max": 12.0,
    "extra_hi": 0.045,
    "extra_blob_hi": 110.0,
}
# Loang tren than chu: blob gon trong vong 2px quanh glyph (sau shift local).
# OK that: ring le mong/cao; loang NG: vet gon 70px+ hoac lap lo chu.
EXTRA_RING_MIN_A = 70
EXTRA_RING_MIN_DIM = 10
EXTRA_RING_ASPECT = 1.45
EXTRA_FILL_HI = 0.70
EXTRA_FILL_NINK = 80
EXTRA_THIN_RING = 0.44
EXTRA_THIN_NINK = 80
EXTRA_SMEAR_BLOB = 120.0
EXTRA_EXPAND_PAD = 12
FADE_METHOD = "line_v23_punct_shift"
ALIGN_SHIFT_X = 16
ALIGN_SHIFT_Y = 8
LINE_CROP_PAD = 16
WIDTH_SLACK = 5.0
HEIGHT_GLYPH_SLACK = 3.0
SHIFT_HI_CAP = 12.0
TEXT_DESKEW = True
TEXT_RECROP = False
TEXT_ANGLE_TOL = 0.15
TEXT_ANGLE_MAX = 8.0
TEXT_REGION_PAD = 10
TEXT_PROJ_SPAN = 2.5
TEXT_PROJ_STEP = 0.1
SHOW_IMAGE = True
SHOW_MAX_W = 1600
SHOW_MAX_H = 900

# ================= IMAGE UTILS =================
def preprocess(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def calc_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calc_density(binary):
    return np.sum(binary == 0) / binary.size


def extract_largest_contour(binary):
    contours, _ = cv2.findContours(
        255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea) if contours else None


def align_label(image):
    """Detect góc → xoay khung → detect lại → crop tem."""
    rotated, rotation, est = label_pipe.straighten_image(image)
    left, right, top, bottom, debug = label_pipe.detect_label(rotated)
    crop = label_pipe.crop_label(rotated, left, right, top, bottom)
    return {
        "rotated": rotated,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "crop": crop,
        "rotation": float(rotation),
        "est": est,
        "debug": debug,
    }


def match_template(image, template, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tpl = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if template.ndim == 3
        else template
    )
    th, tw = tpl.shape[:2]
    h, w = gray.shape[:2]
    if tw > w or th > h:
        return None, 0.0
    result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    if score < thresh:
        return None, float(score)
    return (int(loc[0]), int(loc[1])), float(score)


def match_template_offset(aligned_crop, template=MASTER, origin=ORIGIN_MASTER, thresh=TEMP_RATE):
    """Match tempt.jpg trên crop tem → (dx, dy) cộng vào CHECK_AREA."""
    loc, score = match_template(aligned_crop, template, thresh)
    if loc is None:
        print(f"    WARN: template score={score:.3f} < {thresh} → offset 0,0")
        return (0, 0), score
    ox, oy = origin
    offset = (int(loc[0] - ox), int(loc[1] - oy))
    print(f"    template score={score:.3f} offset=({offset[0]}, {offset[1]})")
    return offset, score


def crop_check_roi(align, area=CHECK_AREA, offset=(0, 0)):
    """Crop ROI check trên tem đã thẳng + offset template. Bbox trên ảnh đã xoay."""
    ax, ay, aw, ah = area
    dx, dy = offset
    ax, ay = ax + dx, ay + dy
    crop = align["crop"]
    if crop is None or crop.size == 0:
        return None, None
    h, w = crop.shape[:2]
    x1 = max(0, min(int(ax), w - 1))
    y1 = max(0, min(int(ay), h - 1))
    x2 = max(x1 + 1, min(int(ax + aw), w))
    y2 = max(y1 + 1, min(int(ay + ah), h))
    if x2 <= x1 or y2 <= y1:
        return None, None
    roi = crop[y1:y2, x1:x2]
    rx1 = align["left"] + x1
    ry1 = align["top"] + y1
    rx2 = align["left"] + x2
    ry2 = align["top"] + y2
    return roi, (rx1, ry1, rx2, ry2)


# ================= FEATURE EXTRACTION =================
def extract_features(word_roi):
    gray = cv2.cvtColor(word_roi, cv2.COLOR_BGR2GRAY)
    binary = preprocess(gray)
    ink = binary == 0
    ys = np.where(np.any(ink, axis=1))[0]
    xs = np.where(np.any(ink, axis=0))[0]
    ink_h = int(ys[-1] - ys[0] + 1) if len(ys) else 0
    ink_w = int(xs[-1] - xs[0] + 1) if len(xs) else 0
    return {
        "height": ink_h or binary.shape[0],
        "width": ink_w or binary.shape[1],
        "area": int(np.sum(ink)),
        "density": calc_density(binary),
        "sharpness": calc_sharpness(gray),
        "stroke": calc_stroke_width(binary),
        "contour": extract_largest_contour(binary),
        "dirt": count_dirt_blobs(binary),
    }


def calc_stroke_width(binary):
    """Bề rộng nét ≈ 2 * mean distance-transform trên mực."""
    ink = ((binary == 0).astype(np.uint8)) * 255
    if not np.any(ink):
        return 0.0
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    vals = dist[ink > 0]
    if vals.size == 0:
        return 0.0
    return float(2.0 * np.mean(vals))


def count_dirt_blobs(binary):
    """Blob mực nhỏ, tách khỏi nét chữ chính → bẩn."""
    ink = ((binary == 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 1:
        return 0
    areas = [cv2.contourArea(c) for c in contours]
    main = max(areas) if areas else 0
    if main <= 0:
        return 0
    dirt = 0
    for area in areas:
        if 4 <= area < 0.12 * main:
            dirt += 1
    return dirt


# # ================= OCR WORD DETECTION =================
# def split_line_to_words(line_img, line_text, line_box):
#     x1, y1, x2, y2 = line_box

#     gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

#     bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     tokens = line_text.strip().split()

#     if not tokens:
#         return []

#     h, w = bw.shape

#     # =========================================================
#     # 1. VERTICAL PROJECTION
#     # =========================================================

#     col = np.sum(bw > 0, axis=0)

#     # Không dùng 5% nữa.
#     # Chỉ cần có pixel mực là giữ lại.
#     ink = col > 0

#     # =========================================================
#     # 2. TÌM CÁC VÙNG MỰC
#     # =========================================================

#     groups = []
#     start = None

#     for i, v in enumerate(ink):

#         if v and start is None:
#             start = i

#         elif not v and start is not None:
#             groups.append((start, i - 1))
#             start = None

#     if start is not None:
#         groups.append((start, w - 1))

#     if not groups:
#         return []

#     # =========================================================
#     # 3. GAP GIỮA CÁC VÙNG MỰC
#     # =========================================================

#     gaps = []

#     for i in range(len(groups) - 1):

#         gap = groups[i + 1][0] - groups[i][1] - 1

#         gaps.append(gap)

#     # =========================================================
#     # 4. SỐ WORD OCR
#     # =========================================================

#     n_words = len(tokens)

#     # Chỉ có 1 word
#     if n_words == 1:

#         gx1 = groups[0][0]
#         gx2 = groups[-1][1]

#         roi_bw = bw[:, gx1 : gx2 + 1]

#         ys = np.where(np.any(roi_bw > 0, axis=1))[0]

#         if len(ys):
#             gy1 = int(ys[0])
#             gy2 = int(ys[-1])
#         else:
#             gy1 = 0
#             gy2 = h - 1

#         return [
#             {
#                 "text": tokens[0],
#                 "bbox": (
#                     int(x1 + gx1),
#                     int(y1 + gy1),
#                     int(gx2 - gx1 + 1),
#                     int(gy2 - gy1 + 1),
#                 ),
#                 "score": 0,
#                 "polygon": [],
#             }
#         ]

#     # =========================================================
#     # 5. CẦN N - 1 ĐIỂM CẮT
#     # =========================================================

#     if len(gaps) < n_words - 1:
#         return [
#             {
#                 "text": line_text.strip(),
#                 "bbox": (x1, y1, x2 - x1, y2 - y1),
#                 "score": 0,
#                 "polygon": [],
#             }
#         ]

#     # =========================================================
#     # 6. LẤY N-1 GAP LỚN NHẤT
#     # =========================================================

#     split_indices = np.argsort(gaps)[-(n_words - 1) :]

#     split_indices = sorted(split_indices)

#     # =========================================================
#     # 7. CHIA GROUPS THÀNH WORD
#     # =========================================================

#     word_groups = []

#     start_idx = 0

#     for split_idx in split_indices:

#         word_groups.append(groups[start_idx : split_idx + 1])

#         start_idx = split_idx + 1

#     word_groups.append(groups[start_idx:])

#     # =========================================================
#     # 8. KIỂM TRA
#     # =========================================================

#     if len(word_groups) != len(tokens):
#         return [
#             {
#                 "text": line_text.strip(),
#                 "bbox": (x1, y1, x2 - x1, y2 - y1),
#                 "score": 0,
#                 "polygon": [],
#             }
#         ]

#     # =========================================================
#     # 9. TẠO BBOX WORD
#     # =========================================================

#     words = []

#     for token, g in zip(tokens, word_groups):

#         if not g:
#             continue

#         gx1 = g[0][0]
#         gx2 = g[-1][1]

#         roi_bw = bw[:, gx1 : gx2 + 1]

#         ys = np.where(np.any(roi_bw > 0, axis=1))[0]

#         if len(ys):
#             gy1 = int(ys[0])
#             gy2 = int(ys[-1])
#         else:
#             gy1 = 0
#             gy2 = h - 1

#         wx1 = x1 + gx1
#         wy1 = y1 + gy1

#         wx2 = x1 + gx2
#         wy2 = y1 + gy2

#         words.append(
#             {
#                 "text": token,
#                 "bbox": (int(wx1), int(wy1), int(wx2 - wx1 + 1), int(wy2 - wy1 + 1)),
#                 "score": 0,
#                 "polygon": [],
#             }
#         )

#     return words


# def detect_words(image, offset_x=0, offset_y=0):
#     result = ocr.ocr(image)
#     print(type(result))
#     print(type(result[0]) if result else None)
#     words = []

#     if not result:
#         return words

#     for res in result:
#         if res is None:
#             continue

#         # PaddleOCR cũ
#         if hasattr(res, "json"):
#             data = res.json
#             if isinstance(data, str):
#                 data = json.loads(data)

#             data = data.get("res", data)

#             texts = data.get("rec_texts", [])
#             scores = data.get("rec_scores", [])
#             boxes = data.get("rec_polys", data.get("rec_boxes", []))

#         # PaddleOCR trả list
#         elif isinstance(res, list):
#             if not res:
#                 continue

#             texts = []
#             scores = []
#             boxes = []

#             for item in res:
#                 if not item:
#                     continue

#                 box = item[0]
#                 text = item[1][0]
#                 score = item[1][1]

#                 boxes.append(box)
#                 texts.append(text)
#                 scores.append(score)

#         else:
#             continue

#         for i, text in enumerate(texts):
#             text = str(text).strip()
#             if not text:
#                 continue

#             score = float(scores[i]) if i < len(scores) else 0

#             box = np.array(boxes[i], dtype=np.int32)

#             if box.ndim == 2:
#                 x1 = int(np.min(box[:, 0]))
#                 y1 = int(np.min(box[:, 1]))
#                 x2 = int(np.max(box[:, 0]))
#                 y2 = int(np.max(box[:, 1]))
#             else:
#                 x1, y1, x2, y2 = map(int, box[:4])

#             line_roi = image[y1:y2, x1:x2]

#             if line_roi.size == 0:
#                 continue

#             line_words = split_line_to_words(line_roi, text, (x1, y1, x2, y2))

#             for word in line_words:
#                 wx, wy, ww, wh = word["bbox"]

#                 word["bbox"] = (wx + offset_x, wy + offset_y, ww, wh)

#                 word["score"] = score

#                 if word.get("polygon"):
#                     word["polygon"] = [
#                         [p[0] + offset_x, p[1] + offset_y] for p in word["polygon"]
#                     ]

#                 words.append(word)

#     return words


# ================= SHAPE =================
def shape_similarity(contour_ref, contour_test):
    if contour_ref is None or contour_test is None:
        return 0
    score = cv2.matchShapes(contour_ref, contour_test, cv2.CONTOURS_MATCH_I1, 0)
    return max(0.0, 1.0 - score)


# # ================= BUILD REFERENCE =================
# def build_reference(image_folder):
#     db = defaultdict(list)
#     image_folder = Path(image_folder)

#     for image_path in image_folder.glob("*.*"):
#         image = cv2.imread(str(image_path))
#         if image is None:
#             continue

#         # tìm vị trí logo để lấy offset
#         delta_x, delta_y, score = find_logo_delta(image, master_logo)
#         if delta_x is None:
#             print(f"[SKIP] {image_path.name} - logo score={score:.3f}")
#             continue

#         # lấy ROI text lớn
#         x1, y1, x2, y2 = find_text_roi(delta_x, delta_y, image.shape)
#         text_roi = image[y1:y2, x1:x2]

#         # OCR để tách chữ
#         words = detect_words(text_roi, offset_x=x1, offset_y=y1)
#         # lọc chữ rỗng/null
#         valid_words = [
#             w
#             for w in words
#             if w["text"] and w["text"].strip() != "" and w["text"].lower() != "null"
#         ]

#         print(f"[OK] {image_path.name} | logo={score:.3f} | words={len(valid_words)}")

#         # kiểm tra số lượng chữ OCR có đủ không
#         if len(valid_words) < len(words):
#             print(
#                 f"[WARN] {image_path.name}: OCR missing {len(words) - len(valid_words)} words"
#             )

#         # chỉ duyệt valid_words để tránh trùng lặp
#         for item in valid_words:
#             word = item["text"]
#             x, y, w, h = item["bbox"]
#             # x += delta_x
#             # y += delta_y

#             roi = image[y : y + h, x : x + w]
#             if roi.size == 0:
#                 continue

#             feat = extract_features(roi)
#             feat["w"] = w
#             feat["h"] = h
#             feat["x_base"] = x
#             feat["y_base"] = y
#             db[word].append(feat)

#     # tổng hợp spec
#     spec = {}
#     for word, features in db.items():
#         heights = [f["height"] for f in features]
#         widths = [f["width"] for f in features]
#         areas = [f["area"] for f in features]
#         densitys = [f["density"] for f in features]
#         sharpnesses = [f["sharpness"] for f in features]

#         ref_contour = features[0]["contour"]

#         spec[word] = {
#             "x_base": features[0]["x_base"],  # lấy từ ảnh đầu tiên
#             "y_base": features[0]["y_base"],
#             "w": int(np.mean([f["w"] for f in features])),
#             "h": int(np.mean([f["h"] for f in features])),
#             "height_mean": float(np.mean(heights)),
#             "height_std": float(np.std(heights)),
#             "width_mean": float(np.mean(widths)),
#             "width_std": float(np.std(widths)),
#             "area_mean": float(np.mean(areas)),
#             "area_std": float(np.std(areas)),
#             "density_mean": float(np.mean(densitys)),
#             "density_std": float(np.std(densitys)),
#             "sharpness_mean": float(np.mean(sharpnesses)),
#             "sharpness_std": float(np.std(sharpnesses)),
#             "contour": ref_contour.tolist() if ref_contour is not None else None,
#         }

#     print("Spec words:", list(spec.keys()))
#     return spec


# ================= SAVE / LOAD SPEC =================
def save_spec(spec, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)


def load_spec(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if isinstance(spec, dict) and "lines" in spec:
        return spec
    for word in spec:
        if spec[word].get("contour") is not None:
            spec[word]["contour"] = np.array(spec[word]["contour"], dtype=np.int32)
    return spec


def spec_has_lines(spec):
    return isinstance(spec, dict) and bool(spec.get("lines"))


def _expand_range(mean, std, rel, abs_pad, k=4.0, vmin=None, vmax=None):
    """Nguong = mean ± max(k*std, rel*|mean|, abs). Neu co min/max mau OK thi dung chung."""
    mean = float(mean)
    std = float(std or 0.0)
    pad = max(k * std, rel * abs(mean), float(abs_pad))
    if vmin is not None and vmax is not None:
        lo = float(vmin) - pad * 0.5
        hi = float(vmax) + pad * 0.5
    else:
        lo, hi = mean - pad, mean + pad
    return float(lo), float(hi)


def make_slot_thresholds(slot, defaults=None):
    d = dict(DEFAULT_THRESH)
    if defaults:
        d.update(defaults)
    k = float(d["k"])
    area_lo, area_hi = _expand_range(
        slot.get("area_mean", 0),
        slot.get("area_std", 0),
        d["area_rel"],
        d["area_abs"],
        k,
        slot.get("area_min"),
        slot.get("area_max"),
    )
    den_lo, den_hi = _expand_range(
        slot.get("density_mean", 0),
        slot.get("density_std", 0),
        d["density_rel"],
        d["density_abs"],
        k,
        slot.get("density_min"),
        slot.get("density_max"),
    )
    st_lo, st_hi = _expand_range(
        slot.get("stroke_mean", 0),
        slot.get("stroke_std", 0),
        d["stroke_rel"],
        d["stroke_abs"],
        k,
        slot.get("stroke_min"),
        slot.get("stroke_max"),
    )
    sharp_mean = float(slot.get("sharpness_mean") or 0)
    lap_lo = float(slot.get("sharpness_min") or sharp_mean) * float(d["laplacian_ratio"])
    return {
        "area_lo": area_lo,
        "area_hi": area_hi,
        "density_lo": den_lo,
        "density_hi": den_hi,
        "stroke_lo": st_lo,
        "stroke_hi": st_hi,
        "laplacian_lo": lap_lo,
        "check_area": bool(d["check_area"]),
        "check_density": bool(d["check_density"]),
        "check_stroke": bool(d["check_stroke"]),
        "check_laplacian": bool(d["check_laplacian"]),
        "check_template": bool(d["check_template"]),
        "check_ink": bool(d.get("check_ink", False)),
        "check_mark": bool(d.get("check_mark", True)),
        "check_letter": bool(d.get("check_letter", True)),
        "check_fade": bool(d["check_fade"]),
        "check_punct": bool(d["check_punct"]),
    }


def compact_len(text):
    return len(re.sub(r"\s+", "", text or ""))


def punct_len(text):
    return len(re.findall(r"[.,;:!?()]", text or ""))


def line_ocr_counts(text):
    compact = re.sub(r"\s+", "", text or "")
    return {
        "n_chars": len(compact),
        "n_punct": punct_len(compact),
        "n_alnum": len(re.findall(r"[A-Za-z0-9]", compact)),
    }


def make_line_thresholds(line, defaults=None):
    d = dict(DEFAULT_THRESH)
    if defaults:
        d.update(defaults)
    k = float(d["k"])
    h_lo, h_hi = _expand_range(
        line.get("height_mean", 0),
        line.get("height_std", 0),
        d["height_rel"],
        d["height_abs"],
        k,
        line.get("height_min"),
        line.get("height_max"),
    )
    w_lo, w_hi = _expand_range(
        line.get("width_mean", 0),
        line.get("width_std", 0),
        d["width_rel"],
        d["width_abs"],
        k,
        line.get("width_min"),
        line.get("width_max"),
    )
    return {
        "height_lo": h_lo,
        "height_hi": h_hi,
        "width_lo": w_lo,
        "width_hi": w_hi,
        "content_ratio": float(d["content_ratio"]),
        "check_height": bool(d["check_height"]),
        "check_width": bool(d["check_width"]),
        "check_content": bool(d["check_content"]),
        "check_word_count": bool(d["check_word_count"]),
        "check_chars": bool(d["check_chars"]),
        "check_template": bool(d["check_template"]),
        "check_ink": bool(d.get("check_ink", True)),
        "check_mark": bool(d.get("check_mark", True)),
        "check_letter": bool(d.get("check_letter", True)),
        "check_fade": bool(d["check_fade"]),
        "check_punct": bool(d["check_punct"]),
        "chars_lo": int(line.get("n_chars") or compact_len(line.get("text"))),
        "chars_hi": int(line.get("n_chars") or compact_len(line.get("text"))),
        "punct_lo": int(line.get("n_punct") or punct_len(line.get("text"))),
        "punct_hi": int(line.get("n_punct") or punct_len(line.get("text"))),
        "words_lo": int(line.get("n_words") or 0),
        "words_hi": int(line.get("n_words") or 0),
    }


def _merge_keep_user(computed, existing):
    if not isinstance(existing, dict) or not existing:
        return computed
    out = dict(computed)
    out.update(existing)
    return out


def fill_spec_thresholds(spec):
    """Them thresholds vao spec cu, khong de len gia tri user da sua."""
    if not isinstance(spec, dict) or not spec.get("lines"):
        return False
    changed = False
    defaults = dict(DEFAULT_THRESH)
    defaults.update(spec.get("defaults") or {})
    for key in (
        "fade_min",
        "fade_slack",
        "check_fade",
        "check_content",
        "check_word_count",
        "check_chars",
        "check_punct",
        "check_area",
        "check_density",
        "check_stroke",
        "check_laplacian",
        "check_template",
        "check_ink",
        "check_mark",
        "check_letter",
        "check_fade",
        "check_dot",
        "check_shift",
        "check_height",
        "check_width",
        "cover_slack",
        "cover_min",
        "mark_slack",
        "mark_min",
        "letter_slack",
        "letter_min",
        "ink_slack",
        "ink_min",
        "fade_slack",
        "fade_min",
        "match_slack",
        "match_min",
    ):
        defaults[key] = DEFAULT_THRESH[key]
    if spec.get("defaults") != defaults:
        spec["defaults"] = defaults
        changed = True
    n_lines = len(spec.get("lines") or [])
    if int(spec.get("n_lines") or 0) != n_lines:
        spec["n_lines"] = n_lines
        changed = True
    line_force = {
        "check_content": False,
        "check_word_count": False,
        "check_chars": False,
        "check_punct": False,
        "check_template": False,
        "check_ink": True,
        "check_mark": True,
        "check_letter": True,
        "check_fade": True,
        "check_height": True,
        "check_width": True,
        "check_dot": True,
        "check_shift": True,
        "check_extra": True,
    }
    slot_force = {
        "check_area": False,
        "check_density": False,
        "check_stroke": False,
        "check_laplacian": False,
        "check_punct": False,
        "check_template": False,
        "check_ink": False,
        "check_mark": False,
        "check_letter": False,
        "check_fade": False,
    }
    for line in spec["lines"]:
        stats = line_ocr_counts(line.get("text") or " ".join(line.get("tokens") or []))
        if line.get("n_chars") != stats["n_chars"] or line.get("n_punct") != stats["n_punct"]:
            line["n_chars"] = stats["n_chars"]
            line["n_punct"] = stats["n_punct"]
            changed = True
        computed = make_line_thresholds(line, defaults)
        merged = _merge_keep_user(computed, line.get("thresholds"))
        merged.update(line_force)
        hmean = float(line.get("height_mean") or merged.get("height_lo") or 40)
        merged["height_lo"] = round(hmean * 0.70, 2)
        merged["height_hi"] = round(hmean * 1.40, 2)
        merged["glyph_h_slack"] = float(HEIGHT_GLYPH_SLACK)
        nch = int(line.get("n_chars") or stats["n_chars"])
        merged["chars_lo"] = max(1, nch - 1)
        merged["chars_hi"] = nch + 4
        wmean = float(line.get("width_mean") or 0)
        if wmean > 0:
            merged["width_lo"] = float(int(np.floor(wmean - WIDTH_SLACK)))
            merged["width_hi"] = float(int(np.ceil(wmean + WIDTH_SLACK)))
        merged["check_width"] = True
        if "extra_hi" not in merged:
            merged["extra_hi"] = float(DEFAULT_THRESH.get("extra_hi", 0.045))
        if "extra_blob_hi" not in merged:
            merged["extra_blob_hi"] = float(DEFAULT_THRESH.get("extra_blob_hi", 110.0))
        if line.get("thresholds") != merged:
            line["thresholds"] = merged
            changed = True
        for slot in line.get("slots") or []:
            computed_s = make_slot_thresholds(slot, defaults)
            merged_s = _merge_keep_user(computed_s, slot.get("thresholds"))
            merged_s.update(slot_force)
            if "fade_hi" not in merged_s and merged_s.get("fade_mean") is not None:
                merged_s["fade_hi"] = round(float(merged_s["fade_mean"]) + 0.12, 3)
            elif "fade_hi" not in merged_s and merged_s.get("fade_lo") is not None:
                merged_s["fade_hi"] = round(float(merged_s["fade_lo"]) + 0.25, 3)
            if slot.get("thresholds") != merged_s:
                slot["thresholds"] = merged_s
                changed = True
    return changed


_TPL_CACHE = {}


def to_gray(img):
    if img is None or img.size == 0:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def ncc_score(img, templ):
    """NCC 0..1. Can theo chieu cao roi truot ngang — khong bep rong o."""
    aligned, t = align_to_template(img, templ)
    if aligned is None or t is None:
        return 0.0
    g, t = to_gray(aligned), to_gray(t)
    if g is None or t is None or g.shape != t.shape:
        return 0.0
    g = g.astype(np.float32)
    t = t.astype(np.float32)
    g -= g.mean()
    t -= t.mean()
    denom = float(np.sqrt((g * g).sum() * (t * t).sum()))
    if denom < 1e-6:
        return 0.0
    return float(np.clip((g * t).sum() / denom, -1.0, 1.0))


def ink_iou_score(img, templ):
    """Ti le chong muc vs mau — bat mat net."""
    g, t = to_gray(img), to_gray(templ)
    if g is None or t is None or t.size == 0:
        return 0.0
    g = cv2.resize(g, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
    gb = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    tb = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    inter = int(np.logical_and(gb, tb).sum())
    union = int(np.logical_or(gb, tb).sum())
    return float(inter / union) if union else 0.0


def align_to_template(img, templ, pad=8):
    """Can chieu cao voi mau, truot ngang tren crop rong, cat dung cua so mau."""
    g, t, _, _ = align_shift(img, templ, max_x=max(pad, 8), max_y=max(3, pad // 2))
    return g, t


def align_shift(img, templ, max_x=ALIGN_SHIFT_X, max_y=ALIGN_SHIFT_Y):
    """Truot 2D nho. Khong scale rong hang — tranh NCC khoa nham chu khac."""
    g, t = to_gray(img), to_gray(templ)
    if g is None or t is None or t.shape[0] < 4 or t.shape[1] < 4:
        return None, None, 0.0, (0, 0)
    th, tw = t.shape[:2]
    if g.shape[0] < 4 or g.shape[1] < 4:
        return None, None, 0.0, (0, 0)
    gpad = cv2.copyMakeBorder(g, max_y, max_y, max_x, max_x, cv2.BORDER_REPLICATE)
    if gpad.shape[0] < th or gpad.shape[1] < tw:
        gpad = cv2.copyMakeBorder(
            gpad,
            0,
            max(0, th - gpad.shape[0]) + 1,
            0,
            max(0, tw - gpad.shape[1]) + 1,
            cv2.BORDER_REPLICATE,
        )
    res = cv2.matchTemplate(gpad, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    x0, y0 = int(loc[0]), int(loc[1])
    return gpad[y0 : y0 + th, x0 : x0 + tw], t, float(score), (x0 - max_x, y0 - max_y)


def crop_line_window(roi, box, pad=LINE_CROP_PAD):
    """Cat cua so dong + bien de align_shift con cho."""
    if roi is None or roi.size == 0 or not box or len(box) < 4:
        return None, (0, 0)
    x, y, w, h = [int(v) for v in box[:4]]
    rh, rw = roi.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(rw, x + w + pad)
    y2 = min(rh, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0)
    return roi[y1:y2, x1:x2], (x1, y1)


def _sample_dark(gray, drop=16):
    paper = float(np.percentile(gray, 90))
    return (gray < (paper - drop)), paper


def _tittle_mask(tb_b):
    """Cham i / dau cham: cot co net ngan o tren, khe, roi than chu."""
    h, w = tb_b.shape
    mask = np.zeros_like(tb_b, dtype=bool)
    if h < 6 or w < 6:
        return mask
    max_dot = max(4, int(0.24 * h))
    top_lim = int(0.42 * h)
    for x in range(w):
        ys = np.where(tb_b[:, x])[0]
        if ys.size < 2:
            continue
        gaps = np.where(np.diff(ys) > 1)[0]
        starts = np.r_[ys[0], ys[gaps + 1]]
        ends = np.r_[ys[gaps], ys[-1]]
        if len(starts) < 2:
            continue
        y0, y1 = int(starts[0]), int(ends[0])
        if y0 <= top_lim and (y1 - y0 + 1) <= max_dot and int(starts[1]) - y1 >= 2:
            mask[y0 : y1 + 1, x] = True
    if np.any(mask):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
        mask &= tb_b
    cut = max(4, int(0.30 * h))
    body = tb_b.copy()
    body[:cut] = False
    k = max(3, cut - 1)
    body_up = cv2.dilate(body.astype(np.uint8), np.ones((k, 3), np.uint8), 1) > 0
    top = tb_b.copy()
    top[cut:] = False
    top &= ~body_up
    bot_cut = max(h - max(4, int(0.22 * h)), cut + 1)
    body2 = tb_b.copy()
    body2[bot_cut:] = False
    body_dn = cv2.dilate(
        body2.astype(np.uint8), np.ones((max(3, h - bot_cut), 3), np.uint8), 1
    ) > 0
    bot = tb_b.copy()
    bot[:bot_cut] = False
    bot &= ~body_dn
    return mask | top | bot


def _ink_skeleton(tb_b):
    """Truc net mau — de bat gay net / mat net, ke ca chu cuoi dong."""
    sk = np.zeros(tb_b.shape, np.uint8)
    img = (tb_b.astype(np.uint8) * 255)
    ker = np.ones((3, 3), np.uint8)
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, ker)
        sk = cv2.bitwise_or(sk, cv2.subtract(img, cv2.dilate(eroded, ker)))
        img = eroded
    return sk > 0


def _fade_local(g, tb_b, paper, gb_d=None, labels=None, stats=None):
    """Do dam chu lon CON MUC. Cho mat net (hit thap) khong tinh FADE."""
    t_dark = np.clip(paper - g.astype(np.float32), 0, 255)
    if labels is None or stats is None:
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
            tb_b.astype(np.uint8), 8
        )
    else:
        nlab = int(stats.shape[0])
    h = g.shape[0]
    min_h = max(10, int(0.42 * h))
    min_a = max(24, int(0.08 * h * h))
    scores = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_a and bh < min_h:
            continue
        blob = labels == i
        if gb_d is not None and float(gb_d[blob].mean()) < 0.70:
            continue
        scores.append(float(t_dark[blob].mean()) / 255.0)
    if not scores:
        return float(t_dark[tb_b].mean()) / 255.0 if np.any(tb_b) else 0.0
    return float(min(scores))


def letter_fade_scores(g, t, tb_b=None, labels=None, stats=None):
    """Do dam TUNG CHU theo blob mau (khong chia deu, khong cua so 90px).
    Chi bo chu bi cat o mep crop. fade_rel: chu / mau. fade_dip: chu / hang xom."""
    empty = (0.0, 1.0, 1.0, [])
    g, t = to_gray(g), to_gray(t)
    if g is None or t is None or g.shape != t.shape:
        return empty
    if tb_b is None:
        tb_b = _ink_bin(t) > 0
    if not np.any(tb_b):
        return 1.0, 1.0, 1.0, []
    if labels is None or stats is None:
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
            tb_b.astype(np.uint8), 8
        )
    else:
        nlab = int(stats.shape[0])
    paper = float(np.percentile(g, 90))
    t_paper = float(np.percentile(t, 90))
    gb_d = cv2.dilate(_ink_bin(g), np.ones((3, 3), np.uint8), 1) > 0
    sd = np.clip(paper - g.astype(np.float32), 0, 255)
    td = np.clip(t_paper - t.astype(np.float32), 0, 255)
    h, w = g.shape
    min_h = max(10, int(0.42 * h))
    min_a = max(24, int(0.08 * h * h))
    rows = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        if area < min_a and bh < min_h:
            continue
        if bx <= 2 or bx + bw >= w - 2:
            continue
        if _is_punct_blob(area, bw, bh, by, h):
            continue
        if bw <= max(5, int(0.14 * h)):
            continue
        if bh < int(0.50 * h):
            continue
        blob = labels == i
        dx, dy, hit = _best_shift(gb_d, blob)
        if hit < 0.65:
            continue
        fs = _dark_at(g, blob, paper, dx, dy)
        ft = float(td[blob].mean()) / 255.0
        rel = fs / ft if ft > 1e-6 else 1.0
        rows.append([fs, rel, bx + dx, bw, bh, by + dy, 1.0])
    if not rows:
        fade_win = float(sd[tb_b].mean()) / 255.0
        return fade_win, 1.0, 1.0, []
    rows.sort(key=lambda r: r[2])
    rels = [r[1] for r in rows]
    for i, row in enumerate(rows):
        neigh = []
        if i > 0:
            neigh.append(rels[i - 1])
        if i + 1 < len(rels):
            neigh.append(rels[i + 1])
        base = max(neigh) if neigh else 1.0
        row[6] = row[1] / base if base > 1e-6 else 1.0
    fade_win = min(r[0] for r in rows)
    rels_sorted = sorted(r[1] for r in rows)
    fade_rel = rels_sorted[1] if len(rels_sorted) >= 3 else rels_sorted[0]
    dips_sorted = sorted(r[6] for r in rows)
    fade_dip = dips_sorted[1] if len(dips_sorted) >= 3 else dips_sorted[0]
    return fade_win, fade_rel, fade_dip, [tuple(r) for r in rows]


def _best_shift(mask, blob, max_dx=10, max_dy=3, cx=0, cy=0):
    """Offset (dx, dy) cho blob mau trung muc mau nhieu nhat.
    cx, cy: tam search (lech chu lan can), khong phai 0 sau align ca hang."""
    ys, xs = np.where(blob)
    if ys.size == 0:
        return 0, 0, 1.0
    h, w = mask.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    best, bdx, bdy = -1.0, cx, cy
    for dy in range(cy - max_dy, cy + max_dy + 1):
        yy = ys + dy
        y_ok = (yy >= 0) & (yy < h)
        if not y_ok.any():
            continue
        for dx in range(cx - max_dx, cx + max_dx + 1):
            xx = xs + dx
            keep = y_ok & (xx >= 0) & (xx < w)
            if not keep.any():
                continue
            hit = float(mask[yy[keep], xx[keep]].mean())
            if hit > best:
                best, bdx, bdy = hit, dx, dy
                if best >= 0.995:
                    return bdx, bdy, best
    return bdx, bdy, max(float(best), 0.0)


def _shifted_hit(mask, blob, max_dx=10, max_dy=3, cx=0, cy=0):
    """Hit tot nhat khi truot blob mau tren muc mau. Bu lech kerning hang dai."""
    _, _, hit = _best_shift(
        mask, blob, max_dx=max_dx, max_dy=max_dy, cx=cx, cy=cy
    )
    return hit


def _dark_at(gray, blob, paper, dx, dy):
    ys, xs = np.where(blob)
    if ys.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    yy, xx = ys + dy, xs + dx
    keep = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    if not keep.any():
        return 0.0
    return float(
        np.mean(np.clip(paper - gray[yy[keep], xx[keep]].astype(np.float32), 0, 255))
    ) / 255.0


def fade_defect_boxes(rows, rel_lo, dip_lo=None):
    """Box tung chu mo hon nguong."""
    if not rows:
        return []
    boxes = []
    for _fs, rel, x, ww, hh, yy, dip in rows:
        bad = rel_lo is not None and rel < float(rel_lo)
        if dip_lo is not None and dip < float(dip_lo):
            bad = True
        if bad:
            boxes.append((int(x), int(yy), int(ww), int(hh)))
    return boxes


def _is_punct_blob(area, bw, bh, by, h):
    """Cham cau / phay o day hang — khong phai cham i."""
    if h < 8 or area < 2:
        return False
    cy = by + 0.5 * bh
    if cy < 0.48 * h:
        return False
    if bh > 0.40 * h:
        return False
    max_a = max(10, int(0.018 * h * h))
    max_dim = max(7, int(0.34 * h))
    if area > max_a or bw > max_dim or bh > max_dim:
        return False
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    return aspect <= 2.3


def _best_shift_prefer_near(mask, blob, near=6, far=28, min_hit=0.82):
    """Uu tien khop cho hien tai; chi mo rong khi chu that su truot."""
    dx, dy, hit = _best_shift(mask, blob, max_dx=near, max_dy=3)
    if hit >= min_hit:
        return dx, dy, hit
    return _best_shift(mask, blob, max_dx=far, max_dy=6)


def _collect_glyph_shifts(gb_d, labels, stats, h, w, max_dx=28, max_dy=6):
    """Lech tung chu (bo dau cau). Moi hang: cx, dx, dy, hit, bx, by, bw, bh."""
    nlab = int(stats.shape[0])
    min_h = max(10, int(0.42 * h))
    min_a = max(24, int(0.08 * h * h))
    rows = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        if area < 3:
            continue
        if _is_punct_blob(area, bw, bh, by, h):
            continue
        if area < min_a and bh < min_h:
            continue
        blob = labels == i
        dx, dy, hit = _best_shift_prefer_near(gb_d, blob)
        rows.append((bx + 0.5 * bw, dx, dy, hit, bx, by, bw, bh))
    rows.sort(key=lambda r: r[0])
    return rows


def _local_shift(rows, x):
    """Lech chu ben trai (gan dau cau) — residual sau align ca hang."""
    if not rows:
        return 0, 0
    left = [r for r in rows if r[0] <= x + 1.0]
    use = left[-4:] if left else rows[:4]
    dx = int(round(float(np.median([r[1] for r in use]))))
    dy = int(round(float(np.median([r[2] for r in use]))))
    return dx, dy


def _shift_from_rows(rows):
    """Lech chu so voi xu huong ca hang (fit dx theo x). Bat cum truot, bo ramp can hang."""
    if len(rows) < 3:
        return 0.0, []
    xs = np.array([r[0] for r in rows], dtype=np.float64)
    dxs = np.array([r[1] for r in rows], dtype=np.float64)
    if float(np.max(xs) - np.min(xs)) < 8.0:
        fitted = np.full_like(dxs, float(np.median(dxs)))
    else:
        slope, intercept = np.polyfit(xs, dxs, 1)
        fitted = slope * xs + intercept
    resid = np.abs(dxs - fitted)
    shift = float(np.max(resid))
    boxes = []
    for row, res in zip(rows, resid):
        if res >= 8.0:
            _cx, _dx, _dy, _hit, bx, by, bw, bh = row
            boxes.append((int(bx), int(by), int(bw), int(bh)))
    return shift, boxes


def glyph_shift_metric(gb_d, labels, stats, h, w, max_dx=28, max_dy=6):
    """Lech ngang tung chu so voi xu huong hang (sau align ca hang)."""
    rows = _collect_glyph_shifts(gb_d, labels, stats, h, w, max_dx, max_dy)
    return _shift_from_rows(rows)


def _glyph_boxes(labels, stats, h, w, skip_edge=False):
    """BBox chu template (bo dau cau / cham nho)."""
    nlab = int(stats.shape[0])
    min_h = max(10, int(0.42 * h))
    min_a = max(24, int(0.08 * h * h))
    out = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        if area < 3:
            continue
        if _is_punct_blob(area, bw, bh, by, h):
            continue
        if area < min_a and bh < min_h:
            continue
        if skip_edge and (bx <= 3 or bx + bw >= w - 3):
            continue
        out.append((i, bx, by, bw, bh, area))
    return out


def glyph_height_metric(gb, tb, letter_rows=None, labels=None, stats=None):
    """Chieu cao chu: median sample vs template (sau shift local). Bo dau cau / mep."""
    h, w = tb.shape[:2]
    if labels is None or stats is None:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            tb.astype(np.uint8), 8
        )
    hs_t, hs_s = [], []
    boxes = []
    for i, bx, by, bw, bh, _a in _glyph_boxes(
        labels, stats, h, w, skip_edge=True
    ):
        odx, ody = _local_shift(letter_rows or [], bx + 0.5 * bw)
        x0 = max(0, int(bx + odx - 1))
        x1 = min(w, int(bx + bw + odx + 1))
        y0 = max(0, int(by + ody - 4))
        y1 = min(h, int(by + bh + ody + 4))
        if x1 <= x0 or y1 <= y0:
            continue
        band = gb[y0:y1, x0:x1]
        if not np.any(band):
            continue
        ys = np.where(np.any(band, axis=1))[0]
        sh = int(ys.max() - ys.min() + 1)
        hs_t.append(float(bh))
        hs_s.append(float(sh))
        yy0 = y0 + int(ys.min())
        boxes.append((x0, yy0, max(1, x1 - x0), sh, sh - bh))
    n = len(hs_t)
    if n < 3:
        return 0.0, 0.0, 0.0, n, []
    med_s = float(np.median(hs_s))
    med_t = float(np.median(hs_t))
    dh = med_s - med_t
    paint = []
    if abs(dh) >= HEIGHT_GLYPH_SLACK:
        for x, y, bw, bh, dlt in boxes:
            if abs(dlt) >= HEIGHT_GLYPH_SLACK:
                paint.append((int(x), int(y), int(bw), int(bh)))
        if not paint:
            paint = [(int(x), int(y), int(bw), int(bh)) for x, y, bw, bh, _d in boxes]
    return med_s, med_t, dh, n, paint


def _union_box(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _expand_extra_boxes(boxes, letter_boxes, pad=EXTRA_EXPAND_PAD):
    """Gan blob extra vao chu gan nhat — box phu than chu loang, khong chi vet le."""
    if not boxes or not letter_boxes:
        return boxes
    out = []
    for box in boxes:
        x, y, bw, bh = box
        cx, cy = x + 0.5 * bw, y + 0.5 * bh
        best = None
        best_d = 1e9
        for _i, lx, ly, lw, lh, _a in letter_boxes:
            if (
                x - pad >= lx + lw
                or x + bw + pad <= lx
                or y - pad >= ly + lh
                or y + bh + pad <= ly
            ):
                continue
            lcx, lcy = lx + 0.5 * lw, ly + 0.5 * lh
            d = abs(cx - lcx) + abs(cy - lcy)
            if d < best_d:
                best_d = d
                best = (lx, ly, lw, lh)
        out.append(_union_box(box, best) if best is not None else box)
    return out


def _hole_fill(gb, blob, bx, by, bw, bh, h, w, odx, ody):
    """Ty le muc mau lap lo trong chu (o, e, a...)."""
    y0, y1 = max(0, by), min(h, by + bh)
    x0, x1 = max(0, bx), min(w, bx + bw)
    sub = blob[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0, 0
    inv = (~sub).astype(np.uint8)
    ncc, lab, sst, _ = cv2.connectedComponentsWithStats(inv, 4)
    hh, ww = sub.shape
    hole_a = 0
    hole_ink = 0
    for k in range(1, ncc):
        ys, xs = np.where(lab == k)
        if ys.size == 0:
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == hh - 1 or xs.max() == ww - 1:
            continue
        if int(sst[k, cv2.CC_STAT_AREA]) < 6:
            continue
        gys = ys + y0 + ody
        gxs = xs + x0 + odx
        gk = (gys >= 0) & (gys < h) & (gxs >= 0) & (gxs < w)
        if not gk.any():
            continue
        hole_a += int(gk.sum())
        hole_ink += int(gb[gys[gk], gxs[gk]].sum())
    fill = hole_ink / float(hole_a) if hole_a else 0.0
    return fill, hole_ink


def _letter_smear_boxes(gb, labels, stats, letter_rows, h, w):
    """Loang nam tren than chu: blob gon o vong 2px / lap lo / chu mong phinh."""
    boxes = []
    score = 0
    k = np.ones((5, 5), np.uint8)
    for i, bx, by, bw, bh, _area in _glyph_boxes(
        labels, stats, h, w, skip_edge=True
    ):
        blob = labels == i
        odx, ody = _local_shift(letter_rows or [], bx + 0.5 * bw)
        dil = cv2.dilate(blob.astype(np.uint8), k, 1) > 0
        ring = dil & ~blob
        rys, rxs = np.where(ring)
        if rys.size == 0:
            continue
        ryy, rxx = rys + ody, rxs + odx
        rk = (ryy >= 0) & (ryy < h) & (rxx >= 0) & (rxx < w)
        if not rk.any():
            continue
        ink = gb[ryy[rk], rxx[rk]]
        nink = int(ink.sum())
        ov = float(ink.mean())
        ov_img = np.zeros((h, w), np.uint8)
        if nink:
            sel = ink.astype(bool)
            ov_img[ryy[rk][sel], rxx[rk][sel]] = 255
        compact_a = 0
        ncc, _, sst, _ = cv2.connectedComponentsWithStats(ov_img, 8)
        for j in range(1, ncc):
            area = int(sst[j, cv2.CC_STAT_AREA])
            ww = int(sst[j, cv2.CC_STAT_WIDTH])
            hh = int(sst[j, cv2.CC_STAT_HEIGHT])
            min_dim = min(ww, hh)
            if area < EXTRA_RING_MIN_A or min_dim < EXTRA_RING_MIN_DIM:
                continue
            aspect = max(ww, hh) / float(max(min_dim, 1))
            if aspect > EXTRA_RING_ASPECT:
                continue
            if area > compact_a:
                compact_a = area
        fill, hole_ink = _hole_fill(gb, blob, bx, by, bw, bh, h, w, odx, ody)
        thin = bw <= 10 and bh >= int(0.55 * h)
        smear = (
            compact_a > 0
            or (fill >= EXTRA_FILL_HI and hole_ink >= EXTRA_FILL_NINK)
            or (thin and ov >= EXTRA_THIN_RING and nink >= EXTRA_THIN_NINK)
        )
        if not smear:
            continue
        ys, xs = np.where(blob)
        yy, xx = ys + ody, xs + odx
        keep = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        py = list(yy[keep])
        px = list(xx[keep])
        if nink:
            py.extend(ryy[rk][ink.astype(bool)].tolist())
            px.extend(rxx[rk][ink.astype(bool)].tolist())
        if not px:
            continue
        x0, x1 = int(min(px)), int(max(px))
        y0, y1 = int(min(py)), int(max(py))
        boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        score = max(score, compact_a, nink, hole_ink, int(EXTRA_SMEAR_BLOB))
    return boxes, score


def extra_ink_metric(gb, tb, letter_rows=None, labels=None, stats=None):
    """Vet loang/nho muc: blob gon ngoai net + loang tren than chu."""
    h, w = tb.shape[:2]
    tb_d = cv2.dilate(tb.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
    extra = gb & ~tb_d
    ys = np.where(np.any(tb, axis=1))[0]
    if ys.size:
        y1 = max(0, int(ys.min()) - 2)
        y2 = min(h, int(ys.max()) + 3)
        extra[:y1] = False
        extra[y2:] = False
    min_a = max(90, int(0.05 * h * h))
    nlab, _, extra_stats, _ = cv2.connectedComponentsWithStats(
        extra.astype(np.uint8), 8
    )
    boxes = []
    kept = 0
    max_a = 0
    for i in range(1, nlab):
        area = int(extra_stats[i, cv2.CC_STAT_AREA])
        if area < min_a:
            continue
        bw = int(extra_stats[i, cv2.CC_STAT_WIDTH])
        bh = int(extra_stats[i, cv2.CC_STAT_HEIGHT])
        min_dim = min(bw, bh)
        if min_dim < 10:
            continue
        aspect = max(bw, bh) / float(max(min_dim, 1))
        if aspect > 1.85:
            continue
        kept += area
        if area > max_a:
            max_a = area
        boxes.append(
            (
                int(extra_stats[i, cv2.CC_STAT_LEFT]),
                int(extra_stats[i, cv2.CC_STAT_TOP]),
                int(extra_stats[i, cv2.CC_STAT_WIDTH]),
                int(extra_stats[i, cv2.CC_STAT_HEIGHT]),
            )
        )
    if labels is None or stats is None:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            tb.astype(np.uint8), 8
        )
    glyph_boxes = _glyph_boxes(labels, stats, h, w, skip_edge=False)
    boxes = _expand_extra_boxes(boxes, glyph_boxes)
    smear_boxes, smear_score = _letter_smear_boxes(
        gb, labels, stats, letter_rows, h, w
    )
    boxes.extend(smear_boxes)
    if smear_score > max_a:
        max_a = smear_score
    need = max(int(tb.sum()), 1)
    return float(kept) / float(need), float(max_a), boxes


def ink_metrics_pair(g, t, skip_edge=False):
    """Do muc tren cap anh da cung size / da can.
    cover: ty le net mau con muc.
    mark: cham i / dau cau yeu nhat.
    letter: chu/net lon yeu nhat.
    fade_tpl: do dam trung binh ca dong.
    fade_win / fade_rel / fade_dip: chu mo nhat (so mau / so hang xom)."""
    empty = {
        "cover": 0.0,
        "mark": 1.0,
        "letter": 0.0,
        "fade_tpl": 0.0,
        "fade_local": 0.0,
        "fade_win": 0.0,
        "fade_rel": 0.0,
        "fade_dip": 0.0,
        "skel": 0.0,
        "punct": 1.0,
        "punct_n": 0,
        "shift": 0.0,
        "extra": 0.0,
        "extra_blob": 0.0,
        "punct_boxes": [],
        "shift_boxes": [],
        "extra_boxes": [],
        "glyph_h": 0.0,
        "glyph_h_tpl": 0.0,
        "glyph_dh": 0.0,
        "glyph_n": 0,
        "height_boxes": [],
    }
    if g is None or t is None:
        return empty
    g, t = to_gray(g), to_gray(t)
    if g is None or t is None or g.shape != t.shape:
        return empty
    tb = (_ink_bin(t) > 0).astype(np.uint8)
    if not np.any(tb):
        return {
            "cover": 1.0,
            "mark": 1.0,
            "letter": 1.0,
            "fade_tpl": 1.0,
            "fade_local": 1.0,
            "fade_win": 1.0,
            "fade_rel": 1.0,
            "fade_dip": 1.0,
            "skel": 1.0,
            "punct": 1.0,
            "punct_n": 0,
            "shift": 0.0,
            "extra": 0.0,
            "extra_blob": 0.0,
            "punct_boxes": [],
            "shift_boxes": [],
            "extra_boxes": [],
            "glyph_h": 0.0,
            "glyph_h_tpl": 0.0,
            "glyph_dh": 0.0,
            "glyph_n": 0,
            "height_boxes": [],
        }
    paper = float(np.percentile(g, 90))
    gb = _ink_bin(g) > 0
    gb_d = cv2.dilate(gb.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
    tb_b = tb > 0
    cover = float((gb_d & tb_b).sum() / tb_b.sum())
    t_dark = np.clip(paper - g.astype(np.float32), 0, 255)
    fade_tpl = float(np.mean(t_dark[tb_b])) / 255.0
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(tb, 8)
    fade_local = _fade_local(
        g, tb_b, paper, gb_d=gb_d, labels=labels, stats=stats
    )
    fade_win, fade_rel, fade_dip, _ = letter_fade_scores(
        g, t, tb_b=tb_b, labels=labels, stats=stats
    )
    h, w = tb.shape
    letter_rows = _collect_glyph_shifts(gb_d, labels, stats, h, w)
    max_a = max(18, int(0.012 * h * w), int(0.22 * h * h))
    max_dim = max(8, int(0.38 * h))
    marks, letters, skels, puncts = [], [], [], []
    punct_boxes = []
    sk_all = _ink_skeleton(tb_b)
    min_h = max(10, int(0.42 * h))
    min_a = max(24, int(0.08 * h * h))
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        if area < 2:
            continue
        clipped = bx <= 2 or bx + bw >= w - 2
        if skip_edge and clipped:
            continue
        blob = labels == i
        if _is_punct_blob(area, bw, bh, by, h):
            odx, ody = _local_shift(letter_rows, bx + 0.5 * bw)
            hit = _shifted_hit(
                gb, blob, max_dx=4, max_dy=3, cx=odx, cy=ody
            )
            puncts.append(hit)
            if hit < 0.40:
                punct_boxes.append((int(bx), int(by), int(bw), int(bh)))
            continue
        if area < 3:
            continue
        hit = _shifted_hit(gb_d, blob)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        small = area <= max_a and bh <= max_dim and bw <= max_dim and aspect <= 2.6
        if small:
            marks.append(hit)
        elif area >= min_a or bh >= min_h:
            letters.append(hit)
            if not clipped:
                sk = sk_all & blob
                if np.any(sk):
                    skels.append(_shifted_hit(gb_d, sk, max_dx=10, max_dy=3))
    tittle = _tittle_mask(tb_b)
    if np.any(tittle):
        marks.append(_shifted_hit(gb_d, tittle, max_dx=6, max_dy=4))
    if len(letter_rows) < 3:
        shift, shift_boxes = 0.0, []
    else:
        shift, shift_boxes = _shift_from_rows(letter_rows)
    extra, extra_blob, extra_boxes = extra_ink_metric(
        gb, tb_b, letter_rows, labels, stats
    )
    glyph_h, glyph_h_tpl, glyph_dh, glyph_n, height_boxes = glyph_height_metric(
        gb, tb_b, letter_rows, labels, stats
    )
    return {
        "cover": cover,
        "mark": min(marks) if marks else 1.0,
        "letter": min(letters) if letters else cover,
        "fade_tpl": fade_tpl,
        "fade_local": fade_local,
        "fade_win": fade_win,
        "fade_rel": fade_rel,
        "fade_dip": fade_dip,
        "skel": min(skels) if skels else 1.0,
        "punct": min(puncts) if puncts else 1.0,
        "punct_n": len(puncts),
        "shift": shift,
        "extra": extra,
        "extra_blob": extra_blob,
        "punct_boxes": punct_boxes,
        "shift_boxes": shift_boxes,
        "extra_boxes": extra_boxes,
        "glyph_h": glyph_h,
        "glyph_h_tpl": glyph_h_tpl,
        "glyph_dh": glyph_dh,
        "glyph_n": glyph_n,
        "height_boxes": height_boxes,
    }


def slot_ink_metrics(img, templ):
    """Can truot nho roi do muc. Khong truot dai theo chieu ngang hang."""
    g, t, ncc, _ = align_shift(img, templ)
    empty = {
        "cover": 0.0,
        "mark": 1.0,
        "letter": 0.0,
        "fade_tpl": 0.0,
        "fade_local": 0.0,
        "fade_win": 0.0,
        "fade_rel": 0.0,
        "fade_dip": 0.0,
        "skel": 0.0,
        "ncc": 0.0,
    }
    if g is None:
        return empty
    met = ink_metrics_pair(g, t, skip_edge=False)
    met["ncc"] = ncc
    return met


def defect_regions(g, t, min_area=6):
    """O mat muc tren toa do anh da can (cung size mau)."""
    g, t = to_gray(g), to_gray(t)
    if g is None or t is None or g.shape != t.shape:
        return []
    tb_b = _ink_bin(t) > 0
    gb = _ink_bin(g) > 0
    gb_d = cv2.dilate(gb.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
    miss = (tb_b & ~gb_d).astype(np.uint8)
    sk_miss = (_ink_skeleton(tb_b) & ~gb_d).astype(np.uint8)
    if np.any(sk_miss):
        miss = cv2.bitwise_or(miss, sk_miss)
    if not np.any(miss):
        return []
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(miss, 8)
    boxes = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        boxes.append(
            (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            )
        )
    return boxes


def ink_strength(img):
    """Do dam muc tren CHINH crop (giay - muc), 0..1. Cao = net; thap = mo/nhat.
    Khong so pixel voi template nen khong bi lech o / lech khung."""
    g = to_gray(img)
    if g is None or g.size < 16:
        return 0.0
    g = cv2.GaussianBlur(g, (3, 3), 0)
    paper = float(np.percentile(g, 90))
    dark_cut = paper - 16.0
    cand = g < dark_cut
    if not np.any(cand):
        return 0.0
    core_cut = float(np.percentile(g[cand], 40))
    core = g <= core_cut
    if int(core.sum()) < 8:
        core = cand
    strength = float(np.mean(np.clip(paper - g[core].astype(np.float32), 0, 255)))
    return strength / 255.0


def fade_ratio(img, templ=None, windows=1):
    """Diem dam muc. windows=0: cua so truot tren dong, lay o yeu nhat."""
    g = to_gray(img)
    if g is None:
        return 0.0
    n = int(windows) if windows is not None else 1
    if n == 1:
        return ink_strength(g)
    w = g.shape[1]
    if n <= 0:
        win = max(28, int(w * 0.16))
        step = max(10, int(w * 0.08))
        if win >= w:
            return ink_strength(g)
        scores = []
        x = 0
        while x + win <= w:
            scores.append(ink_strength(g[:, x : x + win]))
            x += step
        scores.append(ink_strength(g[:, w - win :]))
        return min(scores) if scores else 0.0
    scores = []
    for i in range(n):
        x1 = int(i * w / n)
        x2 = int((i + 1) * w / n)
        if x2 - x1 < 6:
            continue
        scores.append(ink_strength(g[:, x1:x2]))
    return min(scores) if scores else 0.0


PUNCT_RE = re.compile(r"[.,;:]")


def token_has_punct(text):
    return bool(PUNCT_RE.search(text or ""))


def _align_gray(img, templ):
    g, t = to_gray(img), to_gray(templ)
    if g is None or t is None or t.size == 0:
        return None, None
    g = cv2.resize(g, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
    return g, t


def _ink_bin(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def small_mark_boxes(binary):
    """Blob nho giong dau . , : ;"""
    h, w = binary.shape[:2]
    max_a = max(18, int(0.02 * h * w))
    max_dim = max(10, int(0.55 * h))
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8
    )
    boxes = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > max_a:
            continue
        if bh > max_dim or bw > max_dim:
            continue
        boxes.append(
            (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                bw,
                bh,
                area,
            )
        )
    return boxes


def punct_missing_vs_template(crop, templ, overlap_thr=0.28):
    """So blob dau cau tren mau vs anh. Tra ve so dau bi thieu."""
    g, t = _align_gray(crop, templ)
    if g is None:
        return 0, 0
    gb, tb = _ink_bin(g), _ink_bin(t)
    marks = small_mark_boxes(tb)
    if not marks:
        return 0, 0
    missing = 0
    th, tw = t.shape[:2]
    for x, y, bw, bh, _ in marks:
        pad = max(2, bh // 2)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(tw, x + bw + pad), min(th, y + bh + pad)
        need = tb[y1:y2, x1:x2] > 0
        got = gb[y1:y2, x1:x2] > 0
        n_need = int(need.sum())
        if n_need == 0:
            continue
        if int(np.logical_and(need, got).sum()) / n_need < overlap_thr:
            missing += 1
    return missing, len(marks)


def right_ink_ratio(gray, frac=0.22):
    if gray is None or gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    x0 = max(0, int(w * (1.0 - frac)))
    strip = gray[:, x0:]
    if strip.size == 0:
        return 0.0
    return float((_ink_bin(strip) > 0).mean())


def punct_ok(crop, templ=None, token=""):
    """Check dau cau tren muc, khong dung OCR."""
    if templ is not None:
        missing, nmark = punct_missing_vs_template(crop, templ)
        if nmark:
            return missing == 0
        if token_has_punct(token):
            g, t = _align_gray(crop, templ)
            if g is None:
                return True
            tr, gr = right_ink_ratio(t), right_ink_ratio(g)
            if tr < 0.015:
                return True
            return gr >= 0.45 * tr
    if token_has_punct(token):
        gray = to_gray(crop)
        return right_ink_ratio(gray) > 0.03
    return True


def resolve_template_path(path):
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def load_spec_template(path):
    if not path:
        return None
    key = str(resolve_template_path(path))
    if key not in _TPL_CACHE:
        _TPL_CACHE[key] = cv2.imread(key)
    return _TPL_CACHE[key]


def templates_ready(spec):
    if not spec_has_lines(spec):
        return False
    for line in spec["lines"]:
        th = line.get("thresholds") or {}
        p = th.get("template")
        box = line.get("roi_box") or th.get("roi_box")
        if (
            not p
            or not resolve_template_path(p).is_file()
            or not box
            or len(box) < 4
            or "mark_lo" not in th
            or "letter_lo" not in th
            or "fade_lo" not in th
            or "fade_rel_lo" not in th
            or "fade_dip_lo" not in th
            or "skel_lo" not in th
            or "punct_lo" not in th
            or "shift_hi" not in th
            or th.get("fade_method") != FADE_METHOD
        ):
            return False
    return True


def _ng_names():
    if not NG_FOLDER.is_dir():
        return set()
    return {
        p.name
        for p in NG_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def collect_ok_views(ok_folder):
    views = []
    ok_folder = Path(ok_folder)
    skip = _ng_names()
    for image_path in sorted(ok_folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if image_path.name in skip:
            print(f"[TPL] skip NG {image_path.name}")
            continue
        try:
            image = label_pipe.read_image(image_path)
            align = align_label(image)
        except Exception as exc:
            print(f"[SKIP] {image_path.name} {exc}")
            continue
        offset, _ = match_template_offset(align["crop"])
        text_roi, box = crop_check_roi(align, CHECK_AREA, offset)
        if text_roi is None:
            continue
        work, deskew, _ = prepare_text_roi(text_roi)
        lines = detect_lines(work, 0, 0, use_ocr=False)
        views.append(
            {
                "path": image_path,
                "text_roi": work,
                "lines": lines,
                "roi_origin": box,
                "snapped": None,
                "deskew": deskew,
            }
        )
        print(
            f"[TPL] {image_path.name} lines={len(lines)} "
            f"text_goc={deskew.get('angle', 0):+.3f} deg"
        )
    return views


def build_templates_from_ok(spec, ok_folder=None):
    """Luu anh mau OK + nguong match_lo / ink_lo / fade_lo cho tung hang va tung o."""
    defaults = dict(DEFAULT_THRESH)
    defaults.update((spec or {}).get("defaults") or {})
    views = collect_ok_views(ok_folder or OK_FOLDER)
    if not views or not spec_has_lines(spec):
        print("[TPL] khong du anh OK de tao template")
        return spec
    master = max(
        views,
        key=lambda d: (len(d["lines"]), sum(ln["width"] for ln in d["lines"])),
    )
    SPEC_TEMPL_DIR.mkdir(parents=True, exist_ok=True)
    slack = float(defaults["match_slack"])
    ink_slack = float(defaults["ink_slack"])
    fade_slack = float(defaults["fade_slack"])
    match_min = float(defaults["match_min"])
    ink_min = float(defaults["ink_min"])
    fade_min = float(defaults["fade_min"])
    cover_min = float(defaults.get("cover_min", 0.62))
    cover_slack = float(defaults.get("cover_slack", 0.08))
    mark_min = float(defaults.get("mark_min", 0.30))
    mark_slack = float(defaults.get("mark_slack", 0.12))
    letter_min = float(defaults.get("letter_min", 0.52))
    letter_slack = float(defaults.get("letter_slack", 0.10))
    print(f"[TPL] master={master['path'].name} dir={SPEC_TEMPL_DIR}")
    for ref in spec["lines"]:
        mline = next((x for x in master["lines"] if x["line_index"] == ref["line_index"]), None)
        if mline is not None:
            ref["roi_box"] = [int(v) for v in mline["bbox"][:4]]
    for data in views:
        data["snapped"] = snap_spec_boxes(data["text_roi"], spec["lines"])

    for ref in spec["lines"]:
        li = ref["line_index"]
        mline = next((x for x in master["lines"] if x["line_index"] == li), None)
        if mline is None:
            continue
        snapped_m = (master.get("snapped") or {}).get(li) or [int(v) for v in mline["bbox"][:4]]
        mx, my, mw, mh = [int(v) for v in snapped_m[:4]]
        mroi = master["text_roi"]
        if my + mh > mroi.shape[0] or mx + mw > mroi.shape[1]:
            continue
        mcrop = mroi[my : my + mh, mx : mx + mw]
        if mcrop.size == 0:
            continue
        rel_line = f"spec_templates/L{li}.png"
        cv2.imwrite(str(SPEC_TEMPL_DIR / f"L{li}.png"), mcrop)
        roi_box = [mx, my, mw, mh]
        ref["roi_box"] = roi_box
        line_ncc, covers, marks, letters, fades, fade_rels, fade_dips, skels = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        puncts, shifts, extras, extra_blobs = [], [], [], []

        for data in views:
            snapped = (data.get("snapped") or {}).get(li) or roi_box
            crop, _ = crop_line_window(data["text_roi"], snapped)
            if crop is None or crop.size == 0:
                continue
            g, t, ncc, _ = align_shift(crop, mcrop)
            if g is None:
                continue
            met = ink_metrics_pair(g, t, skip_edge=False)
            line_ncc.append(ncc)
            covers.append(met["cover"])
            marks.append(met["mark"])
            letters.append(met["letter"])
            fades.append(met["fade_tpl"])
            fade_rels.append(met["fade_rel"])
            fade_dips.append(met["fade_dip"])
            skels.append(met["skel"])
            if int(met.get("punct_n") or 0) > 0:
                puncts.append(met["punct"])
            shifts.append(met.get("shift", 0.0))
            extras.append(met.get("extra", 0.0))
            extra_blobs.append(met.get("extra_blob", 0.0))

        th = ref.setdefault("thresholds", {})
        th["template"] = rel_line
        th["roi_box"] = roi_box
        th["check_template"] = False
        th["check_ink"] = True
        th["check_mark"] = True
        th["check_letter"] = True
        th["check_fade"] = True
        th["check_dot"] = True
        th["check_shift"] = True
        th["check_extra"] = True
        th["check_width"] = True
        th["fade_method"] = FADE_METHOD
        th["fade_windows"] = 0
        if line_ncc:
            th["match_lo"] = max(match_min, float(min(line_ncc)) - slack)
            th["match_mean"] = round(float(np.mean(line_ncc)), 3)
        if covers:
            floor = float(np.percentile(covers, 10))
            th["ink_lo"] = round(max(cover_min, floor - cover_slack), 3)
            th["ink_mean"] = round(float(np.mean(covers)), 3)
        if marks:
            robust = [x for x in marks if x >= 0.25] or list(marks)
            th["mark_lo"] = round(min(float(min(robust)) - 0.08, 0.88), 3)
            th["mark_mean"] = round(float(np.mean(marks)), 3)
            th["mark_min_ok"] = round(float(min(marks)), 3)
        if letters:
            robust = [x for x in letters if x >= 0.25] or list(letters)
            th["letter_lo"] = round(float(min(robust)) - 0.04, 3)
            th["letter_mean"] = round(float(np.mean(letters)), 3)
            th["letter_min_ok"] = round(float(min(letters)), 3)
        if fades:
            floor = float(np.percentile(fades, 10))
            th["fade_lo"] = round(max(fade_min, floor - fade_slack), 3)
            th["fade_hi"] = round(min(1.0, float(max(fades)) + fade_slack), 3)
            th["fade_mean"] = round(float(np.mean(fades)), 3)
            th["fade_min_ok"] = round(float(min(fades)), 3)
        if fade_rels:
            robust = [x for x in fade_rels if x >= 0.25] or list(fade_rels)
            th["fade_rel_lo"] = round(float(min(robust)) - 0.04, 3)
            th["fade_rel_mean"] = round(float(np.mean(fade_rels)), 3)
            th["fade_rel_min_ok"] = round(float(min(fade_rels)), 3)
        if fade_dips:
            robust = [x for x in fade_dips if x >= 0.25] or list(fade_dips)
            th["fade_dip_lo"] = round(float(min(robust)) - 0.04, 3)
            th["fade_dip_mean"] = round(float(np.mean(fade_dips)), 3)
            th["fade_dip_min_ok"] = round(float(min(fade_dips)), 3)
        if skels:
            robust = [x for x in skels if x >= 0.25] or list(skels)
            th["skel_lo"] = round(float(min(robust)) - 0.04, 3)
            th["skel_mean"] = round(float(np.mean(skels)), 3)
            th["skel_min_ok"] = round(float(min(skels)), 3)
        if puncts:
            th["punct_lo"] = round(max(0.45, float(min(puncts)) - 0.12), 3)
            th["punct_mean"] = round(float(np.mean(puncts)), 3)
            th["punct_min_ok"] = round(float(min(puncts)), 3)
        else:
            th["punct_lo"] = 0.45
            th["punct_n_tpl"] = 0
        if shifts:
            th["shift_hi"] = round(max(8.0, float(np.percentile(shifts, 90)) + 4.0), 2)
            th["shift_mean"] = round(float(np.mean(shifts)), 2)
            th["shift_max_ok"] = round(float(max(shifts)), 2)
        if extras:
            th["extra_hi"] = round(
                max(0.045, float(np.percentile(extras, 90)) + 0.012), 3
            )
            th["extra_mean"] = round(float(np.mean(extras)), 3)
            th["extra_max_ok"] = round(float(max(extras)), 3)
        if extra_blobs:
            th["extra_blob_hi"] = round(
                max(110.0, float(np.percentile(extra_blobs, 90)) + 24.0), 1
            )
            th["extra_blob_max_ok"] = round(float(max(extra_blobs)), 1)
        print(
            f"  L{li} ink_lo={th.get('ink_lo', 0):.3f} "
            f"mark_lo={th.get('mark_lo', 0):.3f} "
            f"letter_lo={th.get('letter_lo', 0):.3f} "
            f"skel_lo={th.get('skel_lo', 0):.3f} "
            f"fade_lo={th.get('fade_lo', 0):.3f} "
            f"fade_rel_lo={th.get('fade_rel_lo', 0):.3f} "
            f"fade_dip_lo={th.get('fade_dip_lo', 0):.3f} "
            f"punct_lo={th.get('punct_lo', 0):.3f} "
            f"shift_hi={th.get('shift_hi', 0):.1f} "
            f"extra_hi={th.get('extra_hi', 0):.3f} "
            f"extra_blob_hi={th.get('extra_blob_hi', 0):.0f} n={len(line_ncc)}"
        )
        for slot in ref.get("slots") or []:
            sth = slot.setdefault("thresholds", {})
            sth["check_template"] = False
            sth["check_ink"] = False
            sth["check_mark"] = False
            sth["check_letter"] = False
            sth["check_fade"] = False
            sth["check_laplacian"] = False
    return spec


def ensure_spec(spec_file=None, ok_folder=None, build=None, force=None):
    """Build spec tu folder OK neu chua co; co roi thi load."""
    spec_file = Path(spec_file or SPEC_FILE)
    ok_folder = Path(ok_folder or OK_FOLDER)
    build = BUILD_SPEC if build is None else build
    force = FORCE_BUILD_SPEC if force is None else force
    existing = None
    if spec_file.exists():
        try:
            existing = load_spec(spec_file)
        except Exception as exc:
            print(f"[SPEC] khong doc duoc {spec_file}: {exc}")
            existing = None
    ready = spec_has_lines(existing)
    if force or (build and not ready):
        if not ok_folder.is_dir():
            raise FileNotFoundError(f"OK folder khong ton tai: {ok_folder}")
        skip = _ng_names()
        n_ok = sum(
            1
            for p in ok_folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.name not in skip
        )
        why = "FORCE_BUILD_SPEC" if force else "chua co spec lines"
        print(f"[BUILD SPEC] {why} | folder={ok_folder} | images={n_ok}")
        spec = build_reference(ok_folder)
        if not spec_has_lines(spec):
            raise RuntimeError("build_reference khong ra spec.lines")
        fill_spec_thresholds(spec)
        build_templates_from_ok(spec, ok_folder)
        save_spec(spec, spec_file)
        print(f"[BUILD SPEC] saved {spec_file}")
        return spec
    if ready:
        if fill_spec_thresholds(existing):
            save_spec(existing, spec_file)
            print(f"[SPEC] bo sung thresholds vao {spec_file}")
        if not templates_ready(existing):
            print(f"[TEMPLATE] tao mau tu {ok_folder}")
            build_templates_from_ok(existing, ok_folder)
            save_spec(existing, spec_file)
            print(f"[TEMPLATE] saved {spec_file}")
        print(f"[SPEC] dung file co san {spec_file} (BUILD_SPEC={build})")
        return existing
    raise FileNotFoundError(
        f"Chua co spec hop le tai {spec_file}. Bat BUILD_SPEC=True roi chay lai."
    )


# ================= OCR LINE =================
def bbox_iou(a, b):
    x1, y1, x2, y2 = a
    a1, b1, a2, b2 = b
    ix1, iy1 = max(x1, a1), max(y1, b1)
    ix2, iy2 = min(x2, a2), min(y2, b2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)
    union = area1 + area2 - inter
    return inter / union if union else 0


def same_line(a, b):
    x1, y1, x2, y2 = a
    a1, b1, a2, b2 = b
    overlap = min(y2, b2) - max(y1, b1)
    if overlap <= 0:
        return False
    return overlap / min(y2 - y1, b2 - b1) > 0.6


def filter_lines(lines):
    # Ưu tiên OCR box có confidence cao
    lines = sorted(lines, key=lambda x: x["score"], reverse=True)
    keep = []

    for cur in lines:
        duplicate = False

        for old in keep:
            if bbox_iou(cur["bbox"], old["bbox"]) > 0.3:
                duplicate = True
                break

            if same_line(cur["bbox"], old["bbox"]):
                # Nếu 2 box cùng một dòng và overlap theo chiều ngang
                cx1, cy1, cx2, cy2 = cur["bbox"]
                ox1, oy1, ox2, oy2 = old["bbox"]

                horizontal_overlap = min(cx2, ox2) - max(cx1, ox1)

                if horizontal_overlap > 0:
                    duplicate = True
                    break

        if not duplicate:
            keep.append(cur)

    keep.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return keep


# ================= SPLIT LINES / WORDS BY WHITESPACE =================
def ink_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def identity_deskew(roi):
    h, w = roi.shape[:2]
    return {
        "roi": roi,
        "angle": 0.0,
        "rotation": 0.0,
        "src_box": (0, 0, w, h),
        "recrop": (0, 0, w, h),
        "M": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "ok": True,
    }


def detect_text_region(roi, pad=TEXT_REGION_PAD):
    """Bbox om khoi chu trong ROI (khong dung OCR)."""
    if roi is None or roi.size == 0:
        return None, None
    h, w = roi.shape[:2]
    bw = ink_mask(roi)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    clean = np.zeros(bw.shape, dtype=np.uint8)
    for i in range(1, nlab):
        if int(stats[i, cv2.CC_STAT_AREA]) >= 8:
            clean[labels == i] = 255
    if not np.any(clean):
        clean = (bw > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    blob = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    pts = cv2.findNonZero(blob)
    if pts is None:
        pts = cv2.findNonZero(clean)
    if pts is None:
        return (0, 0, w, h), None
    x, y, bw_, bh_ = cv2.boundingRect(pts)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw_ + pad)
    y2 = min(h, y + bh_ + pad)
    rect = cv2.minAreaRect(pts.reshape(-1, 2).astype(np.float32))
    return (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))), rect


def minarea_skew_deg(rect):
    if rect is None:
        return 0.0
    (_cx, _cy), (rw, rh), angle = rect
    angle = float(angle)
    if rw < rh:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return float(angle)


def projection_skew_deg(bw, center=0.0, span=TEXT_PROJ_SPAN, step=TEXT_PROJ_STEP):
    if bw is None or bw.size == 0:
        return float(center)
    h, w = bw.shape[:2]
    src = (bw > 0).astype(np.uint8) * 255
    best, best_a = -1.0, float(center)
    a = float(center) - float(span)
    hi = float(center) + float(span)
    cx, cy = w / 2.0, h / 2.0
    while a <= hi + 1e-9:
        M = cv2.getRotationMatrix2D((cx, cy), -float(a), 1.0)
        rot = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(axis=1).astype(np.float64)))
        if score > best:
            best, best_a = score, float(a)
        a += step
    return best_a


def estimate_text_angle(roi, rect=None):
    bw = ink_mask(roi)
    a0 = minarea_skew_deg(rect)
    a = projection_skew_deg(bw, center=a0)
    return float(np.clip(a, -TEXT_ANGLE_MAX, TEXT_ANGLE_MAX))


def deskew_text_roi(roi):
    """Detect khoi chu -> do goc rieng -> xoay thang truoc OCR.
    TEXT_RECROP=False: giu canvas sau xoay (hang check on dinh hon crop sat)."""
    if roi is None or roi.size == 0 or not TEXT_DESKEW:
        return identity_deskew(roi)
    src_box, rect = detect_text_region(roi, pad=TEXT_REGION_PAD)
    if src_box is None:
        return identity_deskew(roi)
    x, y, w, h = src_box
    crop = roi[y : y + h, x : x + w]
    if crop.size == 0:
        return identity_deskew(roi)
    _, rect_c = detect_text_region(crop, pad=0)
    angle = estimate_text_angle(crop, rect=rect_c if rect_c is not None else rect)
    rotation = 0.0
    warped = crop
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if abs(angle) >= TEXT_ANGLE_TOL:
        rotation = -float(angle)
        warped, M = label_pipe.rotate_image(crop, rotation)
        M = np.asarray(M, dtype=np.float32)
    if TEXT_RECROP:
        box2, _ = detect_text_region(warped, pad=TEXT_REGION_PAD)
        if box2 is None:
            box2 = (0, 0, warped.shape[1], warped.shape[0])
        x2, y2, w2, h2 = box2
        straight = warped[y2 : y2 + h2, x2 : x2 + w2]
        if straight.size == 0:
            straight = warped
            box2 = (0, 0, warped.shape[1], warped.shape[0])
    else:
        box2 = (0, 0, warped.shape[1], warped.shape[0])
        straight = warped
    print(
        f"    TEXT box=({x},{y},{w},{h}) goc={angle:+.3f} deg "
        f"-> xoay {rotation:+.3f} deg | work={straight.shape[1]}x{straight.shape[0]}"
    )
    return {
        "roi": straight,
        "angle": float(angle),
        "rotation": float(rotation),
        "src_box": (int(x), int(y), int(w), int(h)),
        "recrop": (int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3])),
        "M": M,
        "ok": True,
    }


def map_pts_from_work(pts, deskew):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts
    rx, ry = deskew["recrop"][0], deskew["recrop"][1]
    pts_c = pts.copy()
    pts_c[:, 0] += rx
    pts_c[:, 1] += ry
    Minv = cv2.invertAffineTransform(np.asarray(deskew["M"], dtype=np.float32))
    pts_crop = label_pipe.transform_points(Minv, pts_c)
    ox, oy = deskew["src_box"][0], deskew["src_box"][1]
    pts_crop[:, 0] += ox
    pts_crop[:, 1] += oy
    return pts_crop


def map_work_box(box, deskew, ox=0, oy=0):
    x, y, w, h = [int(v) for v in box[:4]]
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    mapped = map_pts_from_work(pts, deskew)
    mapped[:, 0] += ox
    mapped[:, 1] += oy
    return mapped.astype(np.int32)


def draw_mapped_box(img, box, deskew, ox, oy, color, thickness):
    quad = map_work_box(box, deskew, ox, oy)
    cv2.polylines(img, [quad.reshape(-1, 1, 2)], True, color, thickness)
    xs, ys = quad[:, 0], quad[:, 1]
    return quad, (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))


def prepare_text_roi(raw_roi):
    deskew = deskew_text_roi(raw_roi)
    work = deskew.get("roi")
    if work is None or work.size == 0:
        deskew = identity_deskew(raw_roi)
        work = raw_roi
    return work, deskew, raw_roi


def mask_segments(mask, min_len=1):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    keep = (ends - starts + 1) >= min_len
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def find_line_bands(bw):
    """Dòng chữ: chiếu mực + tách dải quá cao tại thung lũng."""
    h, w = bw.shape[:2]
    row = np.sum(bw > 0, axis=1).astype(np.float32)
    k = 3
    sm = np.convolve(row, np.ones(k) / k, mode="same")
    thr = max(2.0, float(w) * 0.002)
    raw = mask_segments(sm > thr, MIN_LINE_H)
    if not raw:
        return []
    merged = [list(raw[0])]
    for y1, y2 in raw[1:]:
        if y1 - merged[-1][1] <= 1:
            merged[-1][1] = y2
        else:
            merged.append([y1, y2])
    heights = [b - a + 1 for a, b in merged]
    med = float(np.median(heights)) if heights else float(MIN_LINE_H)
    min_h = max(MIN_LINE_H, int(round(0.55 * med)))
    out = []
    for y1, y2 in merged:
        out.extend(split_tall_band(sm, y1, y2, min_h))
    bands = [(int(a), int(b)) for a, b in out if b >= a]
    return expand_bands_to_glyphs(bw, bands, pad=LINE_BOX_PAD)


def expand_bands_to_glyphs(bw, bands, pad=2):
    """Nới dải để ôm nét nhô lên / thò xuống, dừng trước hàng kề."""
    if not bands:
        return []
    h = bw.shape[0]
    row = np.sum(bw > 0, axis=1).astype(np.float32)
    out = []
    for i, (y1, y2) in enumerate(bands):
        prev_lim = out[i - 1][1] + 1 if i else 0
        next_lim = bands[i + 1][0] if i + 1 < len(bands) else h
        top, bot = int(y1), int(y2)
        while top > prev_lim and row[top - 1] > 0:
            top -= 1
        while bot + 1 < next_lim and row[bot + 1] > 0:
            bot += 1
        top = max(0, top - pad)
        bot = min(h - 1, bot + pad)
        out.append([top, bot])
    for i in range(len(out) - 1):
        if out[i][1] >= out[i + 1][0]:
            mid = (out[i][1] + out[i + 1][0]) // 2
            out[i][1] = mid
            out[i + 1][0] = mid + 1
    return [(int(a), int(b)) for a, b in out if b >= a]


def split_tall_band(row, y1, y2, min_h):
    """Tách 2 hàng dính nhau tại chỗ mực thấp nhất."""
    y1, y2 = int(y1), int(y2)
    n = y2 - y1 + 1
    if n < int(1.7 * min_h):
        return [(y1, y2)]
    seg = np.asarray(row[y1 : y2 + 1], dtype=np.float32)
    k = max(3, n // 15)
    if k % 2 == 0:
        k += 1
    sm = np.convolve(seg, np.ones(k) / k, mode="same")
    lo, hi = min_h, n - min_h
    if hi <= lo:
        return [(y1, y2)]
    valley_rel = int(np.argmin(sm[lo:hi])) + lo
    peak = float(np.max(sm))
    valley = float(sm[valley_rel])
    if peak <= 1.0 or valley > 0.28 * peak:
        return [(y1, y2)]
    mid = y1 + valley_rel
    left = split_tall_band(row, y1, mid - 1, min_h)
    right = split_tall_band(row, mid + 1, y2, min_h)
    return left + right


def merge_word_groups(groups, line_h):
    """Gộp dấu , . vào từ; chỉ cắt khoảng trắng lớn."""
    if not groups:
        return []
    gap_thr = max(6, int(round(line_h * MIN_WORD_GAP_FRAC)))
    punct_w = max(4, int(round(line_h * PUNCT_WIDTH_FRAC)))
    merged = [[groups[0][0], groups[0][1]]]
    for x1, x2 in groups[1:]:
        prev = merged[-1]
        gap = x1 - prev[1] - 1
        prev_w = prev[1] - prev[0] + 1
        cur_w = x2 - x1 + 1
        tiny = min(prev_w, cur_w) <= punct_w
        if gap <= gap_thr or (tiny and gap <= gap_thr * 2):
            prev[1] = x2
        else:
            merged.append([x1, x2])
    return [(int(a), int(b)) for a, b in merged]


def word_boxes_on_line(bw_line, y0):
    h, w = bw_line.shape[:2]
    col = np.sum(bw_line > 0, axis=0)
    groups = mask_segments(col > 0, 1)
    groups = merge_word_groups(groups, h)
    boxes = []
    for x1, x2 in groups:
        ys = np.where(np.any(bw_line[:, x1 : x2 + 1] > 0, axis=1))[0]
        if len(ys) == 0:
            continue
        gy1, gy2 = int(ys[0]), int(ys[-1])
        boxes.append((x1, y0 + gy1, x2 - x1 + 1, gy2 - gy1 + 1))
    return boxes


def parse_ocr_result(result):
    items = []
    if not result:
        return items
    for res in result:
        if res is None:
            continue
        texts, scores, boxes = [], [], []
        if hasattr(res, "json"):
            data = res.json
            if isinstance(data, str):
                data = json.loads(data)
            data = data.get("res", data)
            texts = data.get("rec_texts", []) or []
            scores = data.get("rec_scores", []) or []
            boxes = data.get("rec_polys", data.get("rec_boxes", [])) or []
        elif isinstance(res, list):
            for item in res:
                if not item:
                    continue
                boxes.append(item[0])
                texts.append(item[1][0])
                scores.append(item[1][1])
        else:
            continue
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < OCR_CONF:
                continue
            box = np.array(boxes[i], dtype=np.int32)
            if box.ndim == 2:
                x1 = int(np.min(box[:, 0]))
                y1 = int(np.min(box[:, 1]))
                x2 = int(np.max(box[:, 0]))
                y2 = int(np.max(box[:, 1]))
            else:
                x1, y1, x2, y2 = map(int, box[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            items.append({"text": text, "score": score, "bbox": (x1, y1, x2, y2)})
    return items


def ocr_join_text(image):
    items = parse_ocr_result(get_ocr().ocr(image))
    if not items:
        return "", 0.0
    items.sort(key=lambda z: (z["bbox"][1], z["bbox"][0]))
    return " ".join(z["text"] for z in items), float(
        np.mean([z["score"] for z in items])
    )


def split_tokens_keep_punct(text):
    """Tach Rules.Operation / conditions:(1) thanh token dung."""
    t = str(text or "")
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([.,;:!?])([A-Za-z(])", r"\1 \2", t)
    t = re.sub(r"(\))([A-Za-z])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z])(\()", r"\1 \2", t)
    return [p for p in t.split() if p]


def assign_text_to_boxes(line_img, boxes):
    text, score = ocr_join_text(line_img)
    tokens = split_tokens_keep_punct(text)
    if len(tokens) == len(boxes):
        return [(tok, score) for tok in tokens]
    out = []
    for x, y, w, h in boxes:
        pad = 2
        ih, iw = line_img.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)
        crop = line_img[y1:y2, x1:x2]
        t, s = ocr_join_text(crop) if crop.size else ("", 0.0)
        out.append((t, s))
    return out


# ================= DETECT LINES =================
def even_slots(bbox, n):
    """Chia đều bbox hàng thành n ô (không tách blob)."""
    x, y, w, h = bbox
    n = max(int(n), 1)
    slots = []
    for i in range(n):
        x1 = x + int(round(i * w / n))
        x2 = x + int(round((i + 1) * w / n))
        slots.append((int(x1), int(y), max(1, int(x2 - x1)), int(h)))
    return slots


def expand_slot_box(box, line_bbox, frac=0.32):
    """Nới ô sang 2 bên để NCC còn chỗ tìm đúng chữ."""
    x, y, w, h = [int(v) for v in box]
    lx, ly, lw, lh = [int(v) for v in line_bbox]
    pad = max(10, int(frac * w))
    x1 = max(lx, x - pad)
    x2 = min(lx + lw, x + w + pad)
    return (int(x1), int(y), max(1, int(x2 - x1)), int(h))


def fit_boxes_to_n(boxes, n, line_bbox):
    """Gom/tach bbox OCR ve dung n o spec."""
    n = max(int(n), 1)
    boxes = [tuple(int(v) for v in b) for b in boxes]
    if not boxes:
        return even_slots(line_bbox, n)
    boxes = sorted(boxes, key=lambda b: b[0])
    while len(boxes) > n and len(boxes) >= 2:
        gaps = [
            (boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]), i)
            for i in range(len(boxes) - 1)
        ]
        _, i = min(gaps)
        a, b = boxes[i], boxes[i + 1]
        x1 = min(a[0], b[0])
        x2 = max(a[0] + a[2], b[0] + b[2])
        boxes[i : i + 2] = [(x1, a[1], max(1, x2 - x1), a[3])]
    while len(boxes) < n:
        i = max(range(len(boxes)), key=lambda k: boxes[k][2])
        x, y, w, h = boxes[i]
        if w < 10:
            break
        w1 = max(1, w // 2)
        boxes[i : i + 1] = [(x, y, w1, h), (x + w1, y, max(1, w - w1), h)]
    if len(boxes) < n:
        extra = even_slots(line_bbox, n)[len(boxes) :]
        boxes.extend(extra)
    return boxes[:n]


def ocr_slots_on_line(ocr_boxes, n, line_bbox):
    """X,W tu OCR; Y,H lay ca hang de khong cat cham i / dau cau."""
    lx, ly, lw, lh = [int(v) for v in line_bbox]
    fitted = fit_boxes_to_n(ocr_boxes, n, line_bbox)
    out = []
    for i, (x, _, w, _) in enumerate(fitted):
        x1 = max(lx, x - 2)
        x2 = min(lx + lw, x + w + 8)
        if i + 1 < len(fitted):
            x2 = min(x2, max(x1 + 1, fitted[i + 1][0] - 1))
        out.append((int(x1), int(ly), max(1, int(x2 - x1)), int(lh)))
    return out


def norm_tokens(tokens):
    return [str(t).strip().lower() for t in tokens if str(t).strip()]


def content_key(tokens):
    s = " ".join(norm_tokens(tokens))
    s = s.replace("ofthe", "of the")
    s = s.replace("complieswith", "complies with")
    s = s.replace("maynot", "may not")
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def content_match(got_tokens, ref_tokens, min_ratio=CONTENT_RATIO):
    a, b = content_key(got_tokens), content_key(ref_tokens)
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= min_ratio


def group_ocr_into_lines(items):
    """Gom box OCR theo hang (chong lap Y)."""
    items = sorted(items, key=lambda z: (z["bbox"][1], z["bbox"][0]))
    groups = []
    for item in items:
        placed = False
        for group in groups:
            if same_line(item["bbox"], group[0]["bbox"]):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    for group in groups:
        group.sort(key=lambda z: z["bbox"][0])
    groups.sort(key=lambda g: min(z["bbox"][1] for z in g))
    return groups


def ink_bbox_for_band(bw, ly1, ly2, pad=1):
    """Bbox ôm hết mực hàng (kể cả nét nhô/thò), chỉ bỏ hạt rác."""
    h, w = bw.shape[:2]
    ly1 = max(0, int(ly1))
    ly2 = min(h - 1, int(ly2))
    if ly2 < ly1:
        return None
    band = bw[ly1 : ly2 + 1]
    if band.size == 0:
        return None
    ink = (band > 0).astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    clean = np.zeros_like(ink)
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            clean[labels == i] = 1
    if not np.any(clean):
        clean = ink
    row = np.sum(clean, axis=1)
    col = np.sum(clean, axis=0)
    ys = np.where(row > 0)[0]
    xs = np.where(col > 0)[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    x1 = max(0, int(xs[0]) - pad)
    y1 = max(0, ly1 + int(ys[0]) - pad)
    x2 = min(w - 1, int(xs[-1]) + pad)
    y2 = min(h - 1, ly1 + int(ys[-1]) + pad)
    return (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))


def pick_ink_band(bands, oy1, oy2):
    best, best_ov = None, 0
    oh = max(1, oy2 - oy1)
    for ly1, ly2 in bands:
        ov = max(0, min(oy2, ly2) - max(oy1, ly1))
        if ov > best_ov:
            best, best_ov = (ly1, ly2), ov
    if best is None or best_ov < 0.35 * oh:
        return (oy1, oy2)
    return best


def snap_line_box(roi, spec_box, bands=None, bw=None):
    """Box hang = bbox muc that, khong union voi spec (tranh ho trang)."""
    x, y, w, h = [int(v) for v in (spec_box or [0, 0, 1, 1])[:4]]
    if roi is None or roi.size == 0:
        return [x, y, w, h]
    if bw is None:
        bw = ink_mask(roi)
    if bands is None:
        bands = find_line_bands(bw)
    ly1, ly2 = pick_ink_band(bands, y, y + h)
    ink = ink_bbox_for_band(bw, ly1, ly2, pad=LINE_BOX_PAD)
    if ink is None:
        return [x, int(ly1), w, max(1, int(ly2 - ly1 + 1))]
    return [int(v) for v in ink]


def clip_snapped_boxes(boxes_by_li):
    """Hang ke khong de Y chong nhau."""
    if not boxes_by_li:
        return boxes_by_li
    items = sorted(boxes_by_li.items(), key=lambda kv: kv[1][1])
    out = {li: [int(v) for v in box[:4]] for li, box in items}
    order = [li for li, _ in items]
    for i in range(len(order) - 1):
        li, lj = order[i], order[i + 1]
        ax, ay, aw, ah = out[li]
        bx, by, bw, bh = out[lj]
        a_y2 = ay + ah
        b_y2 = by + bh
        if a_y2 >= by:
            mid = (a_y2 + by) // 2
            out[li] = [ax, ay, aw, max(1, mid - ay)]
            by2 = mid + 1
            out[lj] = [bx, by2, bw, max(1, b_y2 - by2)]
    return out


def snap_spec_boxes(roi, spec_lines):
    """Map tung hang spec toi dai muc. Box = bbox muc, sat chu."""
    boxes = []
    for ref in spec_lines:
        raw = ref.get("roi_box") or (ref.get("thresholds") or {}).get("roi_box")
        boxes.append((ref["line_index"], [int(v) for v in (raw or [0, 0, 1, 1])[:4]]))
    if roi is None or roi.size == 0:
        return {li: box for li, box in boxes}
    bw = ink_mask(roi)
    bands = find_line_bands(bw)
    out = {}
    if len(bands) == len(boxes) and bands:
        order = sorted(range(len(boxes)), key=lambda i: boxes[i][1][1])
        for rank, idx in enumerate(order):
            li, spec_box = boxes[idx]
            ly1, ly2 = bands[rank]
            ink = ink_bbox_for_band(bw, ly1, ly2, pad=LINE_BOX_PAD)
            out[li] = [int(v) for v in (ink or spec_box)]
        return clip_snapped_boxes(out)
    for li, spec_box in boxes:
        out[li] = snap_line_box(roi, spec_box, bands=bands, bw=bw)
    return clip_snapped_boxes(out)


def assign_ocr_to_bands(items, bands):
    """Moi box OCR vao dung 1 hang (tam Y, roi overlap)."""
    assigned = [[] for _ in bands]
    if not bands:
        return assigned
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        cy = 0.5 * (y1 + y2)
        hit = None
        for i, (ly1, ly2) in enumerate(bands):
            if ly1 <= cy <= ly2:
                hit = i
                break
        if hit is None:
            best_i, best = 0, -1.0
            for i, (ly1, ly2) in enumerate(bands):
                ov = max(0, min(y2, ly2) - max(y1, ly1))
                if ov > best:
                    best, best_i = ov, i
            hit = best_i
        assigned[hit].append(item)
    return assigned


def detect_lines(image, offset_x=0, offset_y=0, use_ocr=False):
    """Tach hang theo dai muc. OCR chi neu use_ocr=True (khong dung luc inspect/build)."""
    if image is None or image.size == 0:
        return []
    bw = ink_mask(image)
    bands = find_line_bands(bw)
    items = parse_ocr_result(get_ocr().ocr(image)) if use_ocr else []
    assigned = assign_ocr_to_bands(items, bands)
    lines = []
    line_index = 0
    for (ly1, ly2), group in zip(bands, assigned):
        ink = ink_bbox_for_band(bw, ly1, ly2, pad=LINE_BOX_PAD)
        if ink is None:
            continue
        x, y, w, h = ink
        group = sorted(group, key=lambda z: z["bbox"][0])
        tokens, scores = [], []
        ocr_boxes = []
        for z in group:
            tokens.extend(split_tokens_keep_punct(z["text"]))
            scores.append(z["score"])
            x1, y1, x2, y2 = z["bbox"]
            ocr_boxes.append(
                (int(offset_x + x1), int(offset_y + y1), int(x2 - x1), int(y2 - y1))
            )
        if use_ocr and not tokens:
            continue
        bbox = (
            int(offset_x + x),
            int(offset_y + y),
            int(w),
            int(h),
        )
        lines.append(
            {
                "line_index": line_index,
                "text": " ".join(tokens),
                "tokens": tokens,
                "n_words": len(tokens),
                "score": float(np.mean(scores)) if scores else 0.0,
                "bbox": bbox,
                "ocr_boxes": ocr_boxes,
                "height": bbox[3],
                "width": bbox[2],
            }
        )
        line_index += 1
    return lines


def detect_words(image, offset_x=0, offset_y=0):
    """Giữ API cũ: trải slot đều từ detect_lines (spec n_words lúc inspect)."""
    lines = detect_lines(image, offset_x, offset_y)
    words = []
    for line in lines:
        n = max(line["n_words"], 1)
        for wi, (x, y, w, h) in enumerate(even_slots(line["bbox"], n)):
            token = line["tokens"][wi] if wi < len(line["tokens"]) else ""
            words.append(
                {
                    "text": token,
                    "line_index": line["line_index"],
                    "word_index": wi,
                    "bbox": (x, y, w, h),
                    "score": line["score"],
                    "ink_height": h,
                }
            )
    return words


# ================= BUILD SPEC =================
def build_reference(image_folder):
    image_folder = Path(image_folder)
    all_images = []
    skip = _ng_names()
    for image_path in sorted(image_folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if image_path.name in skip:
            print(f"[SKIP] {image_path.name} (NG)")
            continue
        try:
            image = label_pipe.read_image(image_path)
        except Exception as exc:
            print(f"[SKIP] {image_path.name} read={exc}")
            continue
        try:
            align = align_label(image)
        except Exception as exc:
            print(f"[SKIP] {image_path.name} align={exc}")
            continue
        offset, score = match_template_offset(align["crop"])
        text_roi, box = crop_check_roi(align, CHECK_AREA, offset)
        if text_roi is None:
            print(f"[SKIP] {image_path.name} empty CHECK_AREA")
            continue
        work, deskew, _ = prepare_text_roi(text_roi)
        lines = detect_lines(work, 0, 0, use_ocr=False)
        print(
            f"\n[LINES] {image_path.name} | lines={len(lines)} "
            f"text_goc={deskew.get('angle', 0):+.3f} deg"
        )
        for ln in lines:
            print(f"  LINE {ln['line_index']} h={ln['height']} w={ln['width']}")
        all_images.append({"path": image_path, "image": work, "lines": lines})
    if not all_images:
        return {"lines": []}
    master = max(
        all_images,
        key=lambda d: (len(d["lines"]), sum(ln["width"] for ln in d["lines"])),
    )
    reference = master["lines"]
    print("\n========== MASTER LINES ==========")
    print(f"master={master['path'].name} lines={len(reference)}")
    for ln in reference:
        print(f"LINE {ln['line_index']} h={ln['height']} w={ln['width']}")
    print("====================================")
    db_line = defaultdict(list)
    db_slot = defaultdict(list)
    n_ref = len(reference)
    for data in all_images:
        image = data["image"]
        lines = {ln["line_index"]: ln for ln in data["lines"]}
        if len(data["lines"]) != n_ref:
            print(
                f"[SKIP] {data['path'].name} n_lines={len(data['lines'])} != master {n_ref}"
            )
            continue
        missing = False
        for ref_ln in reference:
            li = ref_ln["line_index"]
            got = lines.get(li)
            if got is None:
                print(f"[SKIP] {data['path'].name} missing LINE {li}")
                missing = True
                break
        if missing:
            continue
        for ref_ln in reference:
            li = ref_ln["line_index"]
            ln = lines[li]
            db_line[li].append(
                {"height": ln["height"], "width": ln["width"], "n_words": ln["n_words"]}
            )
            n = max(int(ref_ln.get("n_words") or 0), 1)
            for wi, (x, y, w, h) in enumerate(even_slots(ln["bbox"], n)):
                roi = image[y : y + h, x : x + w]
                if roi.size == 0:
                    continue
                f = extract_features(roi)
                f.update(x_base=x, y_base=y, w=w, h=h)
                db_slot[(li, wi)].append(f)
    spec_lines = []
    for ref_ln in reference:
        li = ref_ln["line_index"]
        L = db_line.get(li, [])
        if not L:
            continue
        Hs = [v["height"] for v in L]
        Ws = [v["width"] for v in L]
        slots = []
        n = max(int(ref_ln.get("n_words") or 0), 1)
        defaults = dict(DEFAULT_THRESH)
        for wi in range(n):
            fs = db_slot.get((li, wi), [])
            if not fs:
                continue
            toks = ref_ln.get("tokens") or []
            token = toks[wi] if wi < len(toks) else ""
            A = [f["area"] for f in fs]
            D = [f["density"] for f in fs]
            S = [f["sharpness"] for f in fs]
            ST = [f["stroke"] for f in fs]
            slot = {
                "word_index": wi,
                "text": token,
                "area_mean": float(np.mean(A)),
                "area_std": float(np.std(A)),
                "area_min": float(np.min(A)),
                "area_max": float(np.max(A)),
                "density_mean": float(np.mean(D)),
                "density_std": float(np.std(D)),
                "density_min": float(np.min(D)),
                "density_max": float(np.max(D)),
                "sharpness_mean": float(np.mean(S)),
                "sharpness_std": float(np.std(S)),
                "sharpness_min": float(np.min(S)),
                "sharpness_max": float(np.max(S)),
                "stroke_mean": float(np.mean(ST)),
                "stroke_std": float(np.std(ST)),
                "stroke_min": float(np.min(ST)),
                "stroke_max": float(np.max(ST)),
                "samples": len(fs),
            }
            slot["thresholds"] = make_slot_thresholds(slot, defaults)
            slots.append(slot)
        line_spec = {
            "line_index": li,
            "text": ref_ln.get("text") or f"L{li}",
            "tokens": ref_ln.get("tokens") or [],
            "n_words": int(ref_ln.get("n_words") or 0),
            "height_mean": float(np.mean(Hs)),
            "height_std": float(np.std(Hs)),
            "height_min": float(np.min(Hs)),
            "height_max": float(np.max(Hs)),
            "width_mean": float(np.mean(Ws)),
            "width_std": float(np.std(Ws)),
            "width_min": float(np.min(Ws)),
            "width_max": float(np.max(Ws)),
            "samples": len(L),
            "slots": slots,
        }
        line_spec["thresholds"] = make_line_thresholds(line_spec, defaults)
        spec_lines.append(line_spec)
    spec = {
        "n_lines": len(spec_lines),
        "defaults": dict(DEFAULT_THRESH),
        "lines": spec_lines,
    }
    print("\n========== SPEC LINES ==========")
    for ln in spec_lines:
        print(
            f"  L{ln['line_index']} n={ln['n_words']} h={ln['height_mean']:.1f} "
            f"w={ln['width_mean']:.1f} slots={len(ln['slots'])}"
        )
    print("================================")
    return spec


# ================= CHECK WORD =================
def _band_ok(value, mean, std, k=3.0, min_std=1e-6):
    lo = mean - k * max(std, min_std)
    hi = mean + k * max(std, min_std)
    return lo <= value <= hi


def check_template_pair(crop, th, fade_windows=1):
    """Tra ve (reasons, ncc, ink, fade)."""
    reasons = []
    ncc, ink, fade = None, None, None
    need_tpl = th.get("check_template") or th.get("check_ink")
    if need_tpl:
        tpl = load_spec_template(th.get("template"))
        if tpl is not None and crop is not None and crop.size:
            ncc = ncc_score(crop, tpl)
            ink = ink_iou_score(crop, tpl)
            if th.get("check_template") and "match_lo" in th and ncc < float(th["match_lo"]):
                reasons.append("TEMPLATE")
            if th.get("check_ink") and "ink_lo" in th and ink < float(th["ink_lo"]):
                reasons.append("INK")
    if th.get("check_fade", True) and crop is not None and crop.size:
        nwin = th.get("fade_windows")
        if nwin is None:
            nwin = fade_windows
        nwin = int(nwin) if nwin is not None else 1
        fade = fade_ratio(crop, windows=nwin)
        fade_lo = th.get("fade_lo")
        fade_hi = th.get("fade_hi")
        if fade_lo is None:
            fade_lo = DEFAULT_THRESH.get("fade_min", 0.22)
        if fade < float(fade_lo) or (fade_hi is not None and fade > float(fade_hi)):
            reasons.append("FADE")
    return reasons, ncc, ink, fade


def check_line_geometry(line, ref, defaults=None, crop=None):
    """OCR dem ky tu + chieu cao dong. Khong dung OCR de bat mat net / dau cau."""
    reasons = []
    th = ref.get("thresholds") or make_line_thresholds(ref, defaults)
    got = line_ocr_counts(line.get("text") or " ".join(line.get("tokens") or []))
    if th.get("check_chars", False):
        clo = int(th.get("chars_lo", ref.get("n_chars") or compact_len(ref.get("text"))))
        chi = int(th.get("chars_hi", clo + 4))
        if got["n_chars"] < clo or got["n_chars"] > chi:
            reasons.append("CHAR_COUNT")
    if th.get("check_punct"):
        plo = int(th.get("punct_lo", ref.get("n_punct") or 0))
        if got["n_punct"] < plo:
            reasons.append("PUNCT")
    if th.get("check_width") and line is not None:
        wmean = float(ref.get("width_mean") or 0)
        if wmean > 0:
            wlo = float(int(np.floor(wmean - WIDTH_SLACK)))
            whi = float(int(np.ceil(wmean + WIDTH_SLACK)))
            if not (wlo <= line["width"] <= whi):
                reasons.append("WIDTH")
    if th.get("check_word_count") and line is not None:
        wlo = int(th.get("words_lo", ref.get("n_words") or 0))
        whi = int(th.get("words_hi", ref.get("n_words") or 0))
        if not (wlo <= int(line["n_words"]) <= whi):
            reasons.append("WORD_COUNT")
    if th.get("check_content") and line is not None:
        ratio = float(th.get("content_ratio", CONTENT_RATIO))
        if not content_match(line["tokens"], ref.get("tokens") or [], min_ratio=ratio):
            reasons.append("CONTENT")
    return reasons, None, None, None


def check_line_ink(crop, ref, defaults=None):
    """So muc voi mau dong sau khi truot 2D nho. Bat mat chu, cham i/dau, nhoa.
    Gia tri cuoi: giay match template hang (load L*.png + align_shift)."""
    th = (ref or {}).get("thresholds") or {}
    t_tpl0 = time.perf_counter()
    tpl = load_spec_template(th.get("template"))
    if crop is None or crop.size == 0:
        return (["EMPTY_LINE"], None, None, None, [], (0, 0), None, None, 0.0)
    if tpl is None:
        return ([], None, None, None, [], (0, 0), None, None, time.perf_counter() - t_tpl0)
    g, t, ncc, shift = align_shift(crop, tpl)
    dt_tpl = time.perf_counter() - t_tpl0
    if g is None:
        return (["EMPTY_LINE"], ncc, 0.0, 0.0, [], shift, None, None, dt_tpl)
    met = ink_metrics_pair(g, t, skip_edge=False)
    ink = met["cover"]
    fade = min(met["fade_tpl"], met.get("fade_win", met["fade_tpl"]))
    reasons = []
    if th.get("check_ink", True) and "ink_lo" in th and ink < float(th["ink_lo"]):
        reasons.append("INK")
    if th.get("check_mark", True) and "mark_lo" in th and met["mark"] < float(th["mark_lo"]):
        reasons.append("MARK")
    if th.get("check_letter", True):
        if "letter_lo" in th and met["letter"] < float(th["letter_lo"]):
            reasons.append("STROKE")
        elif "skel_lo" in th and met.get("skel", 1.0) < float(th["skel_lo"]):
            reasons.append("STROKE")
    if th.get("check_fade", True):
        rel_lo = th.get("fade_rel_lo")
        dip_lo = th.get("fade_dip_lo")
        rel = met.get("fade_rel", 1.0)
        dip = met.get("fade_dip", 1.0)
        rel_cut = min(float(rel_lo) - 0.10, 0.70) if rel_lo is not None else None
        dip_cut = min(float(dip_lo) - 0.12, 0.68) if dip_lo is not None else None
        if rel_cut is not None and rel < rel_cut:
            reasons.append("FADE")
        elif dip_cut is not None and dip < dip_cut:
            reasons.append("FADE")
    if th.get("check_dot", True) and int(met.get("punct_n") or 0) > 0 and "punct_lo" in th:
        plo = min(float(th["punct_lo"]), 0.40)
        if met["punct"] < plo:
            reasons.append("PUNCT")
    if th.get("check_shift", True):
        shi = min(float(th.get("shift_hi", SHIFT_HI_CAP)), SHIFT_HI_CAP)
        if met.get("shift", 0.0) > shi:
            reasons.append("SHIFT")
    if th.get("check_extra", True):
        ehi = float(th.get("extra_hi", DEFAULT_THRESH.get("extra_hi", 0.045)))
        bhi = float(th.get("extra_blob_hi", DEFAULT_THRESH.get("extra_blob_hi", 110.0)))
        if met.get("extra", 0.0) > ehi or met.get("extra_blob", 0.0) > bhi:
            reasons.append("EXTRA")
    if th.get("check_height", True) and int(met.get("glyph_n") or 0) >= 3:
        slack = float(HEIGHT_GLYPH_SLACK)
        if abs(float(met.get("glyph_dh") or 0.0)) >= slack:
            reasons.append("HEIGHT")
    paint_miss = any(r in reasons for r in ("INK", "STROKE", "MARK", "FADE"))
    boxes = defect_regions(g, t) if paint_miss else []
    if "FADE" in reasons:
        _, _, _, rows = letter_fade_scores(g, t)
        rel_lo = th.get("fade_rel_lo")
        dip_lo = th.get("fade_dip_lo")
        rel_cut = min(float(rel_lo) - 0.10, 0.70) if rel_lo is not None else None
        dip_cut = min(float(dip_lo) - 0.12, 0.68) if dip_lo is not None else None
        boxes.extend(fade_defect_boxes(rows, rel_cut, dip_cut))
    if "PUNCT" in reasons:
        boxes.extend(met.get("punct_boxes") or [])
    if "SHIFT" in reasons:
        boxes.extend(met.get("shift_boxes") or [])
    if "EXTRA" in reasons:
        boxes.extend(met.get("extra_boxes") or [])
    if "HEIGHT" in reasons:
        boxes.extend(met.get("height_boxes") or [])
    return reasons, ncc, ink, fade, boxes, shift, g, t, dt_tpl


def check_slot(feat, ref, defaults=None, crop=None):
    """Can o voi mau roi do: mat chu/net, cham i/dau cau, nhoa muc.
    NCC chi de chac dang dung chu — khong dung lam ly do NG (de lech sang)."""
    fail_reason = []
    th = (ref or {}).get("thresholds") or make_slot_thresholds(ref or {}, defaults)
    ncc, ink, fade = None, None, None
    tpl = load_spec_template(th.get("template"))
    if crop is None or crop.size == 0:
        return (["EMPTY_SLOT"], ncc, ink, fade)
    if tpl is not None:
        ncc = ncc_score(crop, tpl)
        met = slot_ink_metrics(crop, tpl)
        ink = met["cover"]
        fade = met["fade_tpl"]
        if th.get("check_ink") and met["cover"] < min(float(th.get("ink_lo", 0.75)), 0.75):
            fail_reason.append("INK")
        if th.get("check_mark", True) and "mark_lo" in th and met["mark"] < float(th["mark_lo"]):
            fail_reason.append("MARK")
        if th.get("check_letter", True) and "letter_lo" in th and met["letter"] < float(th["letter_lo"]):
            fail_reason.append("STROKE")
    if th.get("check_fade", True):
        if fade is None:
            fade = fade_ratio(crop, windows=3)
        fade_lo = th.get("fade_lo")
        if fade_lo is None:
            fade_lo = DEFAULT_THRESH.get("fade_min", 0.22)
        fade_lo = min(float(fade_lo), 0.24)
        if fade < fade_lo:
            fail_reason.append("FADE")
    if th.get("check_laplacian") and "laplacian_lo" in th and feat:
        if feat.get("sharpness", 0) < th["laplacian_lo"]:
            fail_reason.append("BLUR")
    return fail_reason, ncc, ink, fade


def check_word(feat, ref):
    """Tương thích spec cũ (word dict). Spec mới dùng check_slot."""
    fail_reason = []
    h, w = feat["height"], feat["width"]
    area, density = feat["area"], feat["density"]
    sharp = feat["sharpness"]
    if "height_mean" in ref and not _band_ok(h, ref["height_mean"], ref.get("height_std", 0)):
        fail_reason.append("HEIGHT")
    if "width_mean" in ref and not _band_ok(w, ref["width_mean"], ref.get("width_std", 0)):
        fail_reason.append("WIDTH")
    if "area_mean" in ref and not _band_ok(area, ref["area_mean"], ref.get("area_std", 0)):
        fail_reason.append("AREA")
    if "density_mean" in ref and not _band_ok(
        density, ref["density_mean"], ref.get("density_std", 0)
    ):
        fail_reason.append("DENSITY")
    if sharp < ref.get("sharpness_mean", 0) * 0.8:
        fail_reason.append("SHARPNESS")
    return fail_reason, 0.0


# ================= INSPECT DEBUG =================
def inspect(image_path, spec):
    t0 = time.perf_counter()
    t_mark = t0
    times = {}

    def lap(name):
        nonlocal t_mark
        times[name] = time.perf_counter() - t_mark
        t_mark = time.perf_counter()

    image = label_pipe.read_image(image_path)
    lap("read")
    results = []
    defaults = (spec or {}).get("defaults") if isinstance(spec, dict) else None
    try:
        align = align_label(image)
    except Exception as exc:
        blank = image.copy()
        elapsed = time.perf_counter() - t0
        print(f"    TIME ocr={'on' if USE_OCR else 'off'} read={times.get('read', 0):.3f} total={elapsed:.3f}s ALIGN_FAIL")
        return (
            [
                {
                    "result": "FAIL",
                    "reasons": ["ALIGN_FAIL"],
                    "error": str(exc),
                    "seconds": round(elapsed, 3),
                }
            ],
            blank,
        )
    lap("align")
    offset, temp_score = match_template_offset(align["crop"])
    lap("tpl")
    t_roi0 = time.perf_counter()
    text_roi, box = crop_check_roi(align, CHECK_AREA, offset)
    output = align["rotated"].copy()
    image = align["rotated"]
    lap("crop")
    if text_roi is None:
        elapsed = time.perf_counter() - t0
        print(
            f"    TIME ocr={'on' if USE_OCR else 'off'} "
            f"read={times.get('read', 0):.3f} align={times.get('align', 0):.3f} "
            f"tpl={times.get('tpl', 0):.3f} total={elapsed:.3f}s EMPTY_CHECK_AREA"
        )
        return (
            [
                {
                    "result": "FAIL",
                    "reasons": ["EMPTY_CHECK_AREA"],
                    "temp_score": round(temp_score, 3),
                    "seconds": round(elapsed, 3),
                }
            ],
            output,
        )
    rx1, ry1, rx2, ry2 = box
    logo_score = align["rotation"]
    work, deskew, _raw_roi = prepare_text_roi(text_roi)
    lap("deskew")
    work_vis = work.copy()
    text_angle = float(deskew.get("angle") or 0.0)
    text_rot = float(deskew.get("rotation") or 0.0)
    found = detect_lines(work, 0, 0, use_ocr=USE_OCR)
    lap("lines")
    found_map = {ln["line_index"]: ln for ln in found}
    spec_lines = spec.get("lines") if isinstance(spec, dict) else None
    spec_by_id = {r["line_index"]: r for r in (spec_lines or [])}
    walk = spec_lines if spec_lines else found
    snapped_map = snap_spec_boxes(work, spec_lines) if spec_lines else {}
    lap("snap")
    n_exp = int((spec or {}).get("n_lines") or (len(spec_lines) if spec_lines else 0))
    n_got = len(found)
    if n_exp > 0 and n_got != n_exp:
        results.append(
            {
                "key": "lines",
                "result": "FAIL",
                "reasons": ["LINE_COUNT"],
                "expected": n_exp,
                "got": n_got,
            }
        )

    sx0, sy0, sw0, sh0 = deskew.get("src_box") or (0, 0, text_roi.shape[1], text_roi.shape[0])
    cv2.rectangle(
        output,
        (rx1 + sx0, ry1 + sy0),
        (rx1 + sx0 + sw0, ry1 + sy0 + sh0),
        (0, 220, 220),
        2,
    )

    ink_parts = []
    t_ink = 0.0
    t_line_tpl = 0.0
    for item in walk:
        if spec_lines:
            ref = item
            li = ref["line_index"]
            ln = found_map.get(li)
        else:
            ref = None
            ln = item
            li = ln["line_index"]
        roi_box = snapped_map.get(li)
        if not roi_box and ref is not None:
            roi_box = ref.get("roi_box") or (ref.get("thresholds") or {}).get("roi_box")
        if not roi_box and ln is not None:
            roi_box = list(ln["bbox"])
        if roi_box:
            x, y, lw, lh = [int(v) for v in roi_box[:4]]
        elif ln is not None:
            x, y, lw, lh = ln["bbox"]
        else:
            x, y, lw, lh = 0, 0, 1, 1
        _, (abs_x, abs_y, _aw, _ah) = draw_mapped_box(
            output, (x, y, lw, lh), deskew, rx1, ry1, (80, 80, 80), 1
        )

        line_ncc, line_ink, line_fade = None, None, None
        reasons = []
        if not spec_lines:
            cap = f"L{li} n={ln['n_words']} h={lh} w={lw}"
        elif ln is None and not roi_box:
            reasons = ["MISSING_LINE"]
            cap = f"L{li} MISSING"
        else:
            geo_ln = dict(ln) if ln is not None else {}
            geo_ln["height"] = lh
            geo_ln["width"] = lw
            if not USE_OCR and ref is not None:
                geo_ln["text"] = ref.get("text") or geo_ln.get("text") or ""
                geo_ln["tokens"] = ref.get("tokens") or geo_ln.get("tokens") or []
                geo_ln["n_words"] = ref.get("n_words") or geo_ln.get("n_words") or 0
            geo, _, _, _ = check_line_geometry(geo_ln, ref, defaults)
            reasons.extend(geo)
            if "WIDTH" in reasons:
                ww = int(work.shape[1])
                if x <= 2 or (x + lw) >= ww - 3:
                    reasons = [r for r in reasons if r != "WIDTH"]
            crop, origin = crop_line_window(work, roi_box)
            t_line = time.perf_counter()
            vis, line_ncc, line_ink, line_fade, miss, shift, _, _, dt_ltpl = check_line_ink(
                crop, ref, defaults
            )
            dt_line = time.perf_counter() - t_line
            t_ink += dt_line
            t_line_tpl += float(dt_ltpl or 0.0)
            ink_parts.append(f"L{li}={dt_line:.3f}")
            reasons.extend(vis)
            ox, oy = origin
            sx, sy = shift
            for mx, my, mw, mh in miss:
                dbox = (ox + sx + mx, oy + sy + my, mw, mh)
                draw_mapped_box(output, dbox, deskew, rx1, ry1, (0, 0, 255), 2)
                cv2.rectangle(
                    work_vis,
                    (dbox[0], dbox[1]),
                    (dbox[0] + mw, dbox[1] + mh),
                    (0, 0, 255),
                    2,
                )
            exp_c = ref.get("n_chars", compact_len(ref.get("text")))
            if USE_OCR:
                got_c = line_ocr_counts((ln or {}).get("text") or "")
                cap = (
                    f"L{li} c={got_c['n_chars']}/{exp_c} "
                    f"h={lh} w={lw} "
                    f"{' '.join(reasons) if reasons else 'OK'}"
                )
            else:
                cap = (
                    f"L{li} h={lh} w={lw} "
                    f"{' '.join(reasons) if reasons else 'OK'}"
                )
        results.append(
            {
                "key": f"line_{li}",
                "result": "FAIL" if reasons else "PASS",
                "reasons": reasons,
                "text": (ln or {}).get("text") or (ref or {}).get("text"),
                "expected": (ref or {}).get("text"),
                "n_words": (ln or {}).get("n_words") or (ref or {}).get("n_words"),
                "height": lh,
                "width": lw,
                "line_index": li,
                "bbox": [abs_x, abs_y, lw, lh],
                "template": line_ncc,
                "ink": line_ink,
                "fade": line_fade,
            }
        )
        box_color = (0, 0, 255) if reasons else (0, 200, 0)
        draw_mapped_box(output, (x, y, lw, lh), deskew, rx1, ry1, box_color, 2)
        cv2.rectangle(work_vis, (x, y), (x + lw, y + lh), box_color, 2)
        (tw, _th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cap_x = min(output.shape[1] - tw - 8, rx2 + 8)
        cv2.putText(
            output,
            cap,
            (max(8, cap_x), abs_y + max(16, lh // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
        )
    if not spec_lines:
        results.append(
            {
                "result": "FAIL",
                "reasons": ["SPEC_NEED_REBUILD"],
                "hint": "spec.json must have key lines",
                "n_lines": len(found),
            }
        )

    elapsed = time.perf_counter() - t0
    roi_s = time.perf_counter() - t_roi0
    n_show = len(spec_lines) if spec_lines else len(found)
    print(
        f"    TIME ocr={'on' if USE_OCR else 'off'} "
        f"read={times.get('read', 0):.3f} align={times.get('align', 0):.3f} "
        f"tpl_crop={times.get('tpl', 0):.3f} | "
        f"ROI crop={times.get('crop', 0):.3f} deskew={times.get('deskew', 0):.3f} "
        f"lines={times.get('lines', 0):.3f} snap={times.get('snap', 0):.3f} "
        f"tpl_line={t_line_tpl:.3f} ink={max(0.0, t_ink - t_line_tpl):.3f} "
        f"roi={roi_s:.3f}s | total={elapsed:.3f}s"
        f"{'' if n_exp <= 0 else f' | LINE_COUNT {n_got}/{n_exp}'}"
    )
    if ink_parts:
        print("    TIME ink " + " ".join(ink_parts))
    results.append(
        {
            "key": "time",
            "seconds": round(elapsed, 3),
            "result": "INFO",
            "parts": times,
            "ink": round(t_ink, 3),
            "tpl_line": round(t_line_tpl, 3),
            "ink_chk": round(max(0.0, t_ink - t_line_tpl), 3),
            "roi": round(roi_s, 3),
        }
    )
    results.append(
        {
            "key": "deskew",
            "angle": round(text_angle, 4),
            "rotation": round(text_rot, 4),
            "result": "INFO",
        }
    )
    results.append({"key": "work_vis", "image": work_vis, "result": "INFO"})
    roi_ng = any(r.get("result") == "FAIL" for r in results)
    roi_color = (0, 0, 255) if roi_ng else (0, 255, 0)
    cv2.rectangle(output, (rx1, ry1), (rx2, ry2), roi_color, 3)
    cv2.putText(
        output,
        f"ROI TEXT: {'NG' if roi_ng else 'OK'} | {elapsed:.2f}s | "
        f"lines={n_got}/{n_exp or n_show} | "
        f"tem {logo_score:+.3f} deg | chu {text_angle:+.3f}->{text_rot:+.3f} deg | "
        f"tpl {temp_score:.3f}",
        (rx1, max(30, ry1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        roi_color,
        2,
    )
    cv2.putText(
        output,
        f"{'NG' if roi_ng else 'OK'}  {elapsed:.2f}s",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        roi_color,
        3,
    )
    return results, output


def _fit_image(img, max_w=SHOW_MAX_W, max_h=SHOW_MAX_H):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    scale = min(float(max_w) / max(w, 1), float(max_h) / max(h, 1), 1.0)
    if scale >= 0.999:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))


def show_inspect(output, results, title="Result"):
    if not SHOW_IMAGE or output is None or output.size == 0:
        return 0
    vis = _fit_image(output)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, vis.shape[1], vis.shape[0])
    cv2.moveWindow(title, 40, 40)
    cv2.imshow(title, vis)
    work_item = next((r for r in results if r.get("key") == "work_vis" and r.get("image") is not None), None)
    if work_item is not None:
        roi_vis = _fit_image(work_item["image"], SHOW_MAX_W, SHOW_MAX_H)
        cv2.namedWindow("ROI TEXT (thang)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI TEXT (thang)", roi_vis.shape[1], roi_vis.shape[0])
        cv2.moveWindow("ROI TEXT (thang)", 40 + vis.shape[1] + 20, 40)
        cv2.imshow("ROI TEXT (thang)", roi_vis)
    return cv2.waitKey(0)


#     # ================= INSPECT RUN =================
# def inspect(image_path, spec):
#     image = cv2.imread(image_path)

#     if image is None:
#         raise ValueError(f"Cannot read image: {image_path}")

#     output = image.copy()

#     delta_x, delta_y, score = find_logo_delta(image, master_logo)

#     if delta_x is None:
#         return [
#             {
#                 "result": "FAIL",
#                 "reasons": ["LOGO_NOT_FOUND"],
#                 "logo_score": round(score, 3),
#             }
#         ], output

#     rx1, ry1, rx2, ry2 = find_text_roi(delta_x, delta_y, image.shape)

#     text_roi = image[ry1:ry2, rx1:rx2]

#     words = detect_words(text_roi, rx1, ry1)
#     results = []

#     for item in words:
#         word = item["text"]
#         x, y, w, h = item["bbox"]

#         if word not in spec:
#             result = "FAIL"
#             fail_reason = ["UNKNOWN_WORD"]
#             sim = 0
#         else:
#             roi = image[y : y + h, x : x + w]

#             if roi.size == 0:
#                 result = "FAIL"
#                 fail_reason = ["INVALID_ROI"]
#                 sim = 0
#             else:
#                 feat = extract_features(roi)
#                 fail_reason, sim = check_word(feat, spec[word])
#                 result = "PASS" if not fail_reason else "FAIL"

#         if result == "FAIL":
#             polygon = np.array(item["polygon"], dtype=np.int32)
#             cv2.polylines(output, [polygon], True, (0, 0, 255), 2)

#         results.append(
#             {
#                 "word": word,
#                 "result": result,
#                 "shape_similarity": round(sim, 3),
#                 "reasons": fail_reason,
#                 "ocr_score": round(item["score"], 3),
#                 "bbox": item["bbox"],
#             }
#         )

#     has_fail = any(r["result"] == "FAIL" for r in results)

#     roi_color = (0, 0, 255) if has_fail else (0, 255, 0)

#     cv2.rectangle(output, (rx1, ry1), (rx2, ry2), roi_color, 2)

#     return results, output


# ================= MAIN =================
if __name__ == "__main__":
    spec = ensure_spec()
    print("\n========== SPEC LINES ==========")
    if isinstance(spec, dict) and spec.get("lines"):
        for ln in spec["lines"]:
            print(
                f"  L{ln['line_index']} n={ln['n_words']} "
                f"h={ln.get('height_mean', 0):.1f} w={ln.get('width_mean', 0):.1f} "
                f"| {ln.get('text', '')}"
            )
        print(f"\nTOTAL LINES = {spec.get('n_lines', len(spec['lines']))}")
    else:
        print("  spec.json chua co key lines → can build_reference lai")
    print(f"MODE = line deskew | {FADE_METHOD} | TEXT_DESKEW={TEXT_DESKEW} recrop={TEXT_RECROP} | USE_OCR={USE_OCR} | n_lines spec")
    print(f"FOLDER = {TEST_FOLDER}")
    print(f"CHECK_AREA = {CHECK_AREA}")
    print(f"TEMPLATE = {TEMP_PATH} origin={ORIGIN_MASTER} rate={TEMP_RATE}")
    print("================================\n")

    paths = sorted(
        p for p in TEST_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"TEST n={len(paths)} folder={TEST_FOLDER}")
    print(f"OK_FOLDER = {OK_FOLDER}")
    print(f"NG_FOLDER = {NG_FOLDER}\n")
    for image_path in paths:
        results, output = inspect(str(image_path), spec)
        elapsed = next((r.get("seconds") for r in results if r.get("key") == "time"), None)
        if elapsed is None:
            elapsed = next((r.get("seconds") for r in results if "seconds" in r), 0.0)
        fails = [r for r in results if r.get("result") == "FAIL"]
        status = "NG" if fails else "OK"
        print(f"\n===== {image_path.name} =====  {elapsed:.2f}s  {status}  fail={len(fails)}")
        for r in fails:
            extra = ""
            if r.get("ink") is not None:
                extra += f" ink={r['ink']:.2f}"
            if r.get("fade") is not None:
                extra += f" fade={r['fade']:.2f}"
            print(
                f"  {r.get('key', '-')} {r.get('reasons')} "
                f"{r.get('word') or r.get('text') or ''}{extra}"
            )
        key = show_inspect(output, results, title=f"Result {image_path.name}")
        if key == 27:
            break
    cv2.destroyAllWindows()
