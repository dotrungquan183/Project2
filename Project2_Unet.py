import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Xây dựng mô hình U-Net (CNN)
def _downsample_cnn_block(block_input, channel, is_first = False):
    if is_first:
        conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1)(block_input)
        conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1)(conv1)
        return [block_input, conv1, conv2]
    else:
        maxpool = tf.keras.layers.MaxPool2D(pool_size=2)(block_input)
        conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1)(maxpool)
        conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1)(conv1)
        return [maxpool, conv1, conv2]

def _upsample_cnn_block(block_input, block_counterpart, channel, is_last = False):
    uppool1 = tf.keras.layers.Conv2DTranspose(channel, kernel_size=2, strides=2)(block_input)
    shape_input = uppool1.shape[2]
    shape_counterpart = block_counterpart.shape[2]
    crop_size = int((shape_counterpart-shape_input)/2)
    block_counterpart_crop = tf.keras.layers.Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size)))(block_counterpart)
    concat = tf.keras.layers.Concatenate(axis=-1)([block_counterpart_crop, uppool1])
    conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1)(concat)
    conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1)(conv1)
    if is_last:
        conv3 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=1)(conv2)
        return [concat, conv1, conv2, conv3]
    return [uppool1, concat, conv1, conv2]

# Các block downsampling
ds_block1 = _downsample_cnn_block(tf.keras.layers.Input(shape=(572, 572, 1)), channel=64, is_first=True)
ds_block2 = _downsample_cnn_block(ds_block1[-1], channel=128)
ds_block3 = _downsample_cnn_block(ds_block2[-1], channel=256)
ds_block4 = _downsample_cnn_block(ds_block3[-1], channel=512)
ds_block5 = _downsample_cnn_block(ds_block4[-1], channel=1024)

# Các block upsampling
us_block4 = _upsample_cnn_block(ds_block5[-1], ds_block4[-1], channel=512)
us_block3 = _upsample_cnn_block(us_block4[-1], ds_block3[-1], channel=256)
us_block2 = _upsample_cnn_block(us_block3[-1], ds_block2[-1], channel=128)
us_block1 = _upsample_cnn_block(us_block2[-1], ds_block1[-1], channel=64, is_last=True)

# Tạo mô hình
model = tf.keras.models.Model(inputs=ds_block1[0], outputs=us_block1[-1])
model.summary()

# Hàm tải và tiền xử lý ảnh
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh với dạng grayscale
    img_resized = cv2.resize(img, (572, 572))  # Thay đổi kích thước ảnh để phù hợp với đầu vào mô hình
    img_resized = np.expand_dims(img_resized, axis=-1)  # Thêm chiều màu sắc (1 kênh - grayscale)
    img_resized = np.expand_dims(img_resized, axis=0)  # Thêm chiều batch (1 ảnh)
    img_resized = img_resized / 255.0  # Chuẩn hóa giá trị pixel từ [0, 255] về [0, 1]
    return img_resized

# Đường dẫn tới ảnh
image_path = 'icon/Test_Image.png'  # Đường dẫn đến ảnh của bạn

# Tiền xử lý ảnh
preprocessed_img = load_and_preprocess_image(image_path)

# Dự đoán bằng mô hình
predicted_mask = model.predict(preprocessed_img)

# Chuyển đổi ảnh dự đoán thành nhị phân (0 hoặc 1)
predicted_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)  # Áp dụng ngưỡng 0.5

# Hiển thị ảnh gốc và ảnh phân đoạn
img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img_original, (572, 572))  # Thay đổi kích thước ảnh

fg, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_resized, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(predicted_mask, cmap='gray')
ax[1].set_title('Predicted Mask')

plt.show()
