import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# 1. Tải mô hình VGG19 đã huấn luyện trước
pretrain_net = tf.keras.applications.VGG19(include_top=True, weights="imagenet")

# In ra 5 layers shape cuối cùng của mô hình VGG19
for layer in pretrain_net.layers[-5:]:
    print('layer {}: {}'.format(layer.output.name, layer.output.shape))

print('pretrain_net output: ', pretrain_net.output)

# 2. Chọn đầu ra từ một layer nào đó (ví dụ: layer thứ -5)
net = tf.keras.models.Model(
    inputs=pretrain_net.input,
    outputs=pretrain_net.layers[-5].output
)

# 3. Tạo các lớp tích chập 1x1 và chuyển vị
num_classes = 21
S = 32
F = 7

# Tích chập 1x1 trên feature map output
conv2D = tf.keras.layers.Conv2D(num_classes, kernel_size=1)

# Tích chập chuyển vị (transpose convolution)
conv2DTran = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=F, strides=S)

# 4. Khởi tạo mô hình
model = tf.keras.models.Sequential([
    net,
    conv2D,
    conv2DTran
])

model.summary()

# Vẽ kiến trúc mô hình
plot_model(model, show_shapes=True, show_layer_names=True)

# 5. Hàm tiền xử lý ảnh đầu vào
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)  # Đọc ảnh RGB
    img_resized = cv2.resize(img, (224, 224))  # Resize ảnh về kích thước 224x224 (kích thước đầu vào của VGG19)
    img_resized = tf.keras.applications.vgg19.preprocess_input(img_resized)  # Tiền xử lý ảnh theo chuẩn của VGG19
    return np.expand_dims(img_resized, axis=0)  # Thêm chiều batch

# 6. Đường dẫn tới ảnh
image_path = 'icon/Test_Image.png'  # Đường dẫn tới ảnh của bạn

# Tiền xử lý ảnh
preprocessed_img = load_and_preprocess_image(image_path)

# 7. Dự đoán với mô hình
predicted_output = model.predict(preprocessed_img)

# 8. Hiển thị ảnh gốc và output của mô hình
img_original = cv2.imread(image_path)  # Đọc ảnh gốc
img_resized = cv2.resize(img_original, (224, 224))  # Resize ảnh cho phù hợp

fg, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))  # Chuyển BGR -> RGB để hiển thị đúng màu
ax[0].set_title('Original Image')

ax[1].imshow(predicted_output[0, :, :, 0], cmap='jet')  # Hiển thị output, ví dụ hiển thị lớp đầu tiên
ax[1].set_title('Model Output')

plt.show()
