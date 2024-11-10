import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Hàm để tính ngưỡng theo thuật toán đẳng liệu
def ridler_calvard_thresholding(image, max_iterations=1000, tolerance=1e-5):
    theta_prev = np.mean(image)
    for i in range(max_iterations):
        foreground = image[image > theta_prev]
        background = image[image <= theta_prev]
        m_f = np.mean(foreground) if len(foreground) > 0 else 0
        m_b = np.mean(background) if len(background) > 0 else 0
        theta_new = (m_f + m_b) / 2
        if abs(theta_new - theta_prev) < tolerance:
            break
        theta_prev = theta_new
    return theta_new


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon/NaoBo.png", cv2.IMREAD_GRAYSCALE)

# Áp dụng thuật toán đẳng liệu để tìm ngưỡng
threshold = ridler_calvard_thresholding(image)
print(f'Ngưỡng cuối cùng: {threshold}')

# Phân đoạn ảnh sử dụng ngưỡng tìm được
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)


# Chuyển ảnh sang định dạng phù hợp với tkinter
def cv2_to_tk_image(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return ImageTk.PhotoImage(pil_image)


def display_histogram(canvas_frame, image):
    # Tạo một Figure và vẽ histogram với kích thước mong muốn
    fig, ax = plt.subplots(figsize=(6, 2))  # Điều chỉnh kích thước của histogram
    ax.hist(image.ravel(), bins=256, color='black')
    ax.set_xlabel('Mức xám', fontsize=10)
    ax.set_ylabel('Tần số', fontsize=10)
    ax.set_title('Histogram của Ảnh Xám', fontsize=12)
    ax.grid(True)

    # Điều chỉnh khoảng cách bên trong của biểu đồ
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)

    # Hiển thị Figure trên Tkinter Canvas
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()

    # Đặt widget canvas vào giữa frame với các tùy chọn căn lề và mở rộng
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.config(width=800, height=400)  # Điều chỉnh kích thước canvas trực tiếp (nếu cần)
    canvas_widget.place(x=40, y=40)


# Tạo giao diện với tkinter
root = tk.Tk()
root.title("Thuật toán Ridler Calvart")
root.state("zoomed")  # Chế độ toàn màn hình

# Đặt trọng số của các hàng và cột để căn giữa các thành phần
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Hiển thị ảnh gốc ở phía trên bên trái
image_resized = cv2.resize(image, (700, 400))
image_tk = cv2_to_tk_image(cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR))
label_image = tk.Label(root, image=image_tk, text="Ảnh Gốc", compound="top", font=("Arial", 14))
label_image.place(x=100, y=50)

# Hiển thị histogram ở phía trên bên phải
frame_histogram = tk.Frame(root)
frame_histogram.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
display_histogram(frame_histogram, image)

# Hiển thị ảnh phân đoạn ở phía dưới bên trái
segmented_resized = cv2.resize(segmented_image, (700, 400))
segmented_tk = cv2_to_tk_image(cv2.cvtColor(segmented_resized, cv2.COLOR_GRAY2BGR))
label_segmented = tk.Label(root, image=segmented_tk, text=f"Ảnh Phân Đoạn (Ngưỡng T = {threshold})",
                           compound="top", font=("Arial", 14))
label_segmented.place(x=100, y=500)

# Tạo bảng chú thích
frame_legend = tk.Frame(root, bd=1, relief="groove", padx=15, pady=15)
frame_legend.place(x=1000, y=500)

# Thêm tiêu đề "Chú thích" vào đầu bảng chú thích
title_label = tk.Label(frame_legend, text="Chú thích", font=("Arial", 14, "bold"), anchor="w")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

# Căn giữa các thành phần bên trong bảng chú thích
frame_legend.grid_columnconfigure(0, weight=1)
frame_legend.grid_columnconfigure(1, weight=1)

# Ô màu đen cho đối tượng và chú thích
black_box = tk.Label(frame_legend, bg="black", width=5, height=2)
black_box.grid(row=1, column=0, padx=10, pady=10)
label_black = tk.Label(frame_legend, text="Đối tượng", font=("Arial", 12), anchor="w")
label_black.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Ô màu trắng cho nền và chú thích
white_box = tk.Label(frame_legend, bg="white", width=5, height=2)
white_box.grid(row=2, column=0, padx=10, pady=10)
label_white = tk.Label(frame_legend, text="Nền", font=("Arial", 12), anchor="w")
label_white.grid(row=2, column=1, padx=10, pady=10, sticky="w")

# Bắt đầu vòng lặp chính của tkinter
root.mainloop()
