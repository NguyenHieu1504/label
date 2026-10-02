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
# 1. LẤY MÀU NỀN Ở MÉP ẢNH
# ============================================================

def get_border_color(image, border=10):
    """
    Lấy màu trung vị ở vùng mép ảnh.
    Dùng màu này để padding, tránh chèn nền đen/trắng giả.
    """

    h, w = image.shape[:2]

    pixels = np.concatenate([
        image[:border, :, :].reshape(-1, 3),       # trên
        image[-border:, :, :].reshape(-1, 3),      # dưới
        image[:, :border, :].reshape(-1, 3),       # trái
        image[:, -border:, :].reshape(-1, 3),      # phải
    ], axis=0)

    color = np.median(pixels, axis=0).astype(np.uint8)

    return tuple(color.tolist())


# ============================================================
# 2. ĐƯA ẢNH VÀO CANVAS CÙNG KÍCH THƯỚC
# ============================================================

def pad_to_canvas(image, target_h, target_w):
    """
    Padding = thêm nền quanh ảnh để đạt cùng kích thước.
    Không resize nên chữ không bị co/giãn.
    """

    h, w = image.shape[:2]

    bg_color = get_border_color(image)

    canvas = np.full(
        (target_h, target_w, 3),
        bg_color,
        dtype=np.uint8
    )

    # đặt ảnh vào giữa canvas
    x = (target_w - w) // 2
    y = (target_h - h) // 2

    canvas[y:y+h, x:x+w] = image

    return canvas


# ============================================================
# 3. ECC ALIGN
# ============================================================

def ecc_align(master, test):
    """
    ECC = Enhanced Correlation Coefficient.

    Mục tiêu:
    tìm phép dịch + xoay để Test khớp Master tốt nhất.
    """

    # chuyển ảnh xám
    master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # đổi sang float 0 -> 1
    master_gray = master_gray.astype(np.float32) / 255.0
    test_gray = test_gray.astype(np.float32) / 255.0

    # blur nhẹ để giảm nhiễu
    master_gray = cv2.GaussianBlur(master_gray, (5, 5), 0)
    test_gray = cv2.GaussianBlur(test_gray, (5, 5), 0)

    # ma trận ban đầu:
    # không dịch, không xoay
    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    # EUCLIDEAN:
    # cho phép:
    # - dịch X
    # - dịch Y
    # - xoay
    #
    # KHÔNG scale/co giãn
    motion_model = cv2.MOTION_EUCLIDEAN

    # điều kiện dừng
    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        200,
        1e-7
    )

    # chạy ECC
    cc, warp_matrix = cv2.findTransformECC(
        master_gray,
        test_gray,
        warp_matrix,
        motion_model,
        criteria
    )

    h, w = master.shape[:2]

    # Warp = nắn Test về tọa độ Master
    aligned_test = cv2.warpAffine(
        test,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned_test, warp_matrix, cc


# ============================================================
# MAIN
# ============================================================

master = cv2.imread(MASTER_PATH)
test = cv2.imread(TEST_PATH)

if master is None:
    raise ValueError("Không đọc được ảnh Master")

if test is None:
    raise ValueError("Không đọc được ảnh Test")


print("Master size:", master.shape[:2])
print("Test size  :", test.shape[:2])


# ------------------------------------------------------------
# Đưa hai ảnh về cùng kích thước bằng padding
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ECC
# ------------------------------------------------------------

aligned_test, warp_matrix, ecc_score = ecc_align(
    master_canvas,
    test_canvas
)


# ------------------------------------------------------------
# Difference trước và sau ECC
# ------------------------------------------------------------

master_gray = cv2.cvtColor(
    master_canvas,
    cv2.COLOR_BGR2GRAY
)

test_gray = cv2.cvtColor(
    test_canvas,
    cv2.COLOR_BGR2GRAY
)

aligned_gray = cv2.cvtColor(
    aligned_test,
    cv2.COLOR_BGR2GRAY
)


diff_before = cv2.absdiff(
    master_gray,
    test_gray
)

diff_after = cv2.absdiff(
    master_gray,
    aligned_gray
)


# ------------------------------------------------------------
# In thông tin ECC
# ------------------------------------------------------------

dx = warp_matrix[0, 2]
dy = warp_matrix[1, 2]

angle = math.degrees(
    math.atan2(
        warp_matrix[1, 0],
        warp_matrix[0, 0]
    )
)


print("\n========== ECC RESULT ==========")

print(f"ECC score : {ecc_score:.6f}")

print(f"dx        : {dx:.3f} px")
print(f"dy        : {dy:.3f} px")
print(f"rotation  : {angle:.3f} degree")

print("\nWarp matrix:")
print(warp_matrix)

# BLOCK_SIZE = 32
# ============================================================
# FLOWLITE - BƯỚC 1: CHIA MASTER THÀNH BLOCK 32x32
# ============================================================

BLOCK_SIZE = 32

h, w = master_canvas.shape[:2]

blocks = []

# Chỉ lấy các block đủ 32x32.
# Phần dư ở mép phải / mép dưới tạm thời bỏ qua.
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

        block = master_canvas[
            y:y + BLOCK_SIZE,
            x:x + BLOCK_SIZE
        ]

        blocks.append({
            "x": x,
            "y": y,
            "image": block
        })


# ============================================================
# THÔNG TIN BLOCK
# ============================================================

num_block_x = w // BLOCK_SIZE
num_block_y = h // BLOCK_SIZE

remain_x = w % BLOCK_SIZE
remain_y = h % BLOCK_SIZE

print("\n========== FLOWLITE BLOCK INFO ==========")

print(f"Master canvas : {w} x {h}")
print(f"Block size    : {BLOCK_SIZE} x {BLOCK_SIZE}")

print(f"Block theo X  : {num_block_x}")
print(f"Block theo Y  : {num_block_y}")

print(f"Tổng block    : {len(blocks)}")

print(f"Dư bên phải   : {remain_x} px")
print(f"Dư phía dưới  : {remain_y} px")


# ============================================================
# VẼ GRID BLOCK LÊN MASTER
# ============================================================

master_grid = master_canvas.copy()

# Vẽ từng block 32x32
for block_info in blocks:

    x = block_info["x"]
    y = block_info["y"]

    cv2.rectangle(
        master_grid,
        (x, y),
        (
            x + BLOCK_SIZE - 1,
            y + BLOCK_SIZE - 1
        ),
        (0, 0, 255),      # đỏ
        1
    )


# ============================================================
# FIGURE 1 - KẾT QUẢ ECC
# ============================================================

plt.figure(figsize=(16, 8))


plt.subplot(2, 3, 1)
plt.imshow(
    cv2.cvtColor(
        master_canvas,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("1. MASTER")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(
    cv2.cvtColor(
        test_canvas,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("2. TEST trước ECC")
plt.axis("off")


plt.subplot(2, 3, 3)
plt.imshow(
    cv2.cvtColor(
        aligned_test,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("3. TEST sau ECC")
plt.axis("off")


plt.subplot(2, 3, 4)
plt.imshow(
    diff_before,
    cmap="gray"
)
plt.title("4. Difference trước ECC")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(
    diff_after,
    cmap="gray"
)
plt.title("5. Difference sau ECC")
plt.axis("off")


# ============================================================
# OVERLAY MASTER / TEST SAU ECC
# ============================================================

overlay = np.zeros(
    (
        target_h,
        target_w,
        3
    ),
    dtype=np.uint8
)

# đỏ = Master
overlay[:, :, 0] = master_gray

# xanh = Test sau ECC
overlay[:, :, 1] = aligned_gray


plt.subplot(2, 3, 6)
plt.imshow(overlay)
plt.title("6. Overlay Master / Test ECC")
plt.axis("off")

plt.tight_layout()


# ============================================================
# FIGURE 2 - MASTER ĐÃ CHIA BLOCK
# ============================================================

plt.figure(figsize=(20, 6))

plt.imshow(
    cv2.cvtColor(
        master_grid,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"MASTER GRID - Block {BLOCK_SIZE}x{BLOCK_SIZE} "
    f"- {num_block_x} x {num_block_y} "
    f"= {len(blocks)} blocks"
)

plt.axis("off")
plt.tight_layout()


# ============================================================
# CHỈ SHOW 1 LẦN
# ============================================================

plt.show()