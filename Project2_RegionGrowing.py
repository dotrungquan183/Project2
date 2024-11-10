from tkinter import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Định nghĩa lớp Point để lưu tọa độ
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


# Hàm tính độ chênh lệch xám giữa các điểm ảnh
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


# Định nghĩa 8 điểm lân cận để kết nối
def selectConnects():
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                Point(-1, 0)]
    return connects


# Hàm phát triển vùng
def regionGrow(img, seeds, thresh):
    m, n = img.shape
    seedMark = np.zeros([m, n])
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects()
    while len(seedList) > 0:
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= m or tmpY >= n:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


# Tải và xử lý ảnh
img = cv2.imread('icon/NaoBo.png', 0)
seeds = [Point(124, 124), Point(283, 125), Point(407, 151), Point(327, 216)]
img_result = regionGrow(img, seeds, 2.75)


# Chuyển ảnh thành định dạng hiển thị trong tkinter
def cv2_to_tk_image(cv2_image):
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB))
    return ImageTk.PhotoImage(pil_image)


# Hàm hiển thị histogram
def display_histogram(canvas_frame, image):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(image.ravel(), bins=256, color='gray')
    ax.set_xlabel('Mức xám')
    ax.set_ylabel('Tần số')
    ax.set_title('Biểu đồ Histogram')
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().place(x=50, y=10)


# Thiết lập giao diện
root = Tk()
root.title("Phân Đoạn Phát Triển Vùng")
root.state("zoomed")

# Hiển thị ảnh gốc
image_resized = cv2.resize(img, (700, 400))
image_tk = cv2_to_tk_image(image_resized)
label_image = Label(root, image=image_tk, text="Ảnh Gốc", compound="top", font=("Arial", 14))
label_image.place(x=100, y=50)

# Hiển thị ảnh đã phân đoạn
segmented_resized = cv2.resize(img_result.astype(np.uint8) * 255, (700, 400))
segmented_tk = cv2_to_tk_image(segmented_resized)
label_segmented = Label(root, image=segmented_tk, text="Ảnh Phân Đoạn", compound="top", font=("Arial", 14))
label_segmented.place(x=100, y=500)

# Hiển thị histogram trong một frame riêng
frame_histogram = Frame(root, width=1000, height=800)
frame_histogram.place(x=850, y=40)
display_histogram(frame_histogram, img)

# Tạo bảng chú thích
frame_legend = Frame(root, bd=1, relief="groove", padx=15, pady=15)
frame_legend.place(x=900, y=500)

# Thêm tiêu đề "Chú thích" vào đầu bảng chú thích
title_label = Label(frame_legend, text="Chú thích", font=("Arial", 14, "bold"), anchor="w")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

# Căn giữa các thành phần bên trong bảng chú thích
frame_legend.grid_columnconfigure(0, weight=1)
frame_legend.grid_columnconfigure(1, weight=1)

# Ô màu đen cho đối tượng và chú thích
black_box = Label(frame_legend, bg="black", width=5, height=2)
black_box.grid(row=1, column=0, padx=10, pady=10)
label_black = Label(frame_legend, text="Đối tượng", font=("Arial", 12), anchor="w")
label_black.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Ô màu trắng cho nền và chú thích
white_box = Label(frame_legend, bg="white", width=5, height=2)
white_box.grid(row=2, column=0, padx=10, pady=10)
label_white = Label(frame_legend, text="Nền", font=("Arial", 12), anchor="w")
label_white.grid(row=2, column=1, padx=10, pady=10, sticky="w")


# Hàm để cập nhật và hiển thị tọa độ khi di chuột
def show_coordinates(event):
    x, y = event.x, event.y
    coordinates_label.config(text=f"Tọa độ: ({x}, {y})")


# Thêm một label để hiển thị tọa độ
coordinates_label = Label(root, text="Tọa độ: (0, 0)", font=("Arial", 12))
coordinates_label.place(x=100, y=470)

# Gán sự kiện di chuột vào ảnh gốc
label_image.bind("<Motion>", show_coordinates)

root.mainloop()
