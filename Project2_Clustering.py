import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Đường dẫn tới ảnh (có thể thay thế với đường dẫn tuyệt đối nếu ảnh nằm trong thư mục của bạn)
img_path = 'icon/Test_Image.png'

# Kiểm tra xem ảnh có tồn tại không
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found at: {img_path}")

# Đọc ảnh bằng OpenCV
img = cv2.imread(img_path)

# Kiểm tra xem ảnh có được đọc thành công không
if img is None:
    raise ValueError("Failed to load image. Please check the file path.")

# Chuyển ảnh từ BGR sang RGB (OpenCV mặc định là BGR, Matplotlib sử dụng RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape X into tensor2D: (width x height, n_channels)
X = img_rgb.reshape((-1, 3))

# Kmeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# In các trung tâm cụm đã tìm được
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)

# Dự đoán nhãn của các điểm ảnh
pred_label = kmeans.predict(X)

# Reshape pred_label về lại hình ảnh ban đầu
X_img = pred_label.reshape(img.shape[:2])

# Hiển thị hình ảnh gốc và kết quả phân cụm
fg, ax = plt.subplots(1, 2, figsize=(8, 4))
for i, image in enumerate([img_rgb, X_img]):
    ax[i].imshow(image)
    ax[i].axis('off')  # Tắt trục
    if i == 0:
        ax[i].set_title('Original Image')
    else:
        ax[i].set_title('k-Means Clustering Image')

# Hiển thị biểu đồ
plt.show()
