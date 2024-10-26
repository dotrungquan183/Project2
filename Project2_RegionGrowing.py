import numpy as np
import cv2
import matplotlib.pyplot as plt


# Định nghĩa lớp Point để lưu trữ tọa độ của các điểm ảnh (pixel)
# Các điểm này sẽ được chọn làm các điểm "hạt giống" (seeds) để khởi đầu quá trình phát triển vùng
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


# Hàm getGrayDiff để tính sự khác biệt về mức xám giữa hai điểm ảnh
# Được sử dụng để kiểm tra xem sự chênh lệch giữa điểm hiện tại và điểm lân cận có thỏa mãn ngưỡng hay không
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


# Hàm selectConnects để định nghĩa 8 điểm lân cận xung quanh một điểm pixel
# Đây là các điểm trong phạm vi kết nối không gian (spatial connectivity) của điểm ảnh hiện tại
def selectConnects():
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                Point(-1, 0)]
    return connects


# Hàm regionGrow để thực hiện quá trình phát triển vùng (region growing)
# img: ảnh đầu vào
# seeds: danh sách các điểm hạt giống
# thresh: ngưỡng để so sánh mức xám và quyết định mở rộng vùng
def regionGrow(img, seeds, thresh):
    m, n = img.shape
    seedMark = np.zeros([m, n])  # Tạo mảng để đánh dấu các điểm đã được phân vào vùng (đánh nhãn các điểm)
    seedList = []  # Tạo danh sách chứa các điểm hạt giống
    for seed in seeds:
        seedList.append(seed)  # Thêm các điểm hạt giống vào danh sách để chuẩn bị cho quá trình phát triển vùng

    label = 1  # Khởi tạo giá trị nhãn cho các vùng (các vùng khác nhau sẽ có nhãn khác nhau)
    connects = selectConnects()  # Lấy danh sách 8 điểm lân cận để kiểm tra mở rộng vùng

    while (len(seedList) > 0):  # Lặp qua các điểm hạt giống trong danh sách
        currentPoint = seedList.pop(0)  # Lấy điểm hạt giống đầu tiên từ danh sách để mở rộng vùng từ điểm này
        seedMark[currentPoint.x, currentPoint.y] = label  # Đánh dấu nhãn cho điểm này

        for i in range(8):  # Kiểm tra từng điểm lân cận trong 8 điểm lân cận
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            # Kiểm tra nếu điểm lân cận nằm ngoài ảnh (giới hạn của ma trận ảnh)
            if tmpX < 0 or tmpY < 0 or tmpX >= m or tmpY >= n:
                continue

            # Tính sự khác biệt về mức xám giữa điểm hiện tại và điểm lân cận
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

            # Nếu sự khác biệt nhỏ hơn ngưỡng và điểm chưa được đánh nhãn
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                # Đánh dấu điểm lân cận này thuộc về vùng hiện tại và gán nhãn
                seedMark[tmpX, tmpY] = label
                # Thêm điểm lân cận vào danh sách hạt giống để tiếp tục phát triển vùng
                seedList.append(Point(tmpX, tmpY))

    # Trả về ảnh phân đoạn với các vùng được đánh nhãn
    return seedMark


if __name__ == "__main__":
    # Đọc ảnh đầu vào dưới dạng ảnh mức xám
    img = cv2.imread('icon/Test_Image.png', 0)

    # Chọn các điểm hạt giống để khởi đầu quá trình phát triển vùng
    seeds = [Point(10, 10), Point(300, 400), Point(100, 300)]

    # Gọi hàm regionGrow để thực hiện thuật toán tăng vùng, với ngưỡng là 3
    img_result = regionGrow(img, seeds, 3)

    # Tạo không gian vẽ ảnh với tỷ lệ 16:9
    fig = plt.figure(figsize=(16, 9))

    # Tạo 2 vùng vẽ con để hiển thị ảnh gốc và ảnh sau khi phân đoạn
    ax1, ax2 = fig.subplots(1, 2)

    # Hiển thị ảnh gốc
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Ảnh gốc')
    ax1.axis('off')

    # Hiển thị ảnh phân đoạn sau khi áp dụng thuật toán tăng vùng
    ax2.imshow(img_result, cmap='gray')
    ax2.set_title('Ảnh phân đoạn Region Growing')
    ax2.axis('off')

    # Hiển thị cả hai ảnh
    plt.show()
