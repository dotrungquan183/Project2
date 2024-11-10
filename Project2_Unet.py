import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Tạo lớp Keras tùy chỉnh để xử lý tf.shape
class ShapeResizer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ShapeResizer, self).__init__(**kwargs)

    def call(self, inputs):
        block_counterpart, uppool1 = inputs
        # Lấy kích thước của uppool1
        up_shape = tf.shape(uppool1)
        # Resize block_counterpart để có cùng kích thước với uppool1
        block_counterpart_resized = tf.image.resize(block_counterpart, size=(up_shape[1], up_shape[2]))
        return block_counterpart_resized


# Xây dựng mô hình U-Net (CNN)
def _downsample_cnn_block(block_input, channel, is_first=False):
    if is_first:
        conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(block_input)
        conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(conv1)
        return [block_input, conv1, conv2]
    else:
        maxpool = tf.keras.layers.MaxPool2D(pool_size=2)(block_input)
        conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(maxpool)
        conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(conv1)
        return [maxpool, conv1, conv2]


def _upsample_cnn_block(block_input, block_counterpart, channel, is_last=False):
    # Upsample using Conv2DTranspose
    uppool1 = tf.keras.layers.Conv2DTranspose(channel, kernel_size=2, strides=2, padding='same')(block_input)

    # Sử dụng lớp tùy chỉnh ShapeResizer để resize
    block_counterpart_resized = ShapeResizer()(inputs=[block_counterpart, uppool1])

    # Nối các tensor sau khi resize
    concat = tf.keras.layers.Concatenate(axis=-1)([block_counterpart_resized, uppool1])

    # Tiếp tục với các lớp Conv2D
    conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(concat)
    conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(conv1)

    if is_last:
        conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv2)
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
