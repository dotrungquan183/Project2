import SimpleITK as sitk
import matplotlib.pyplot as plt

# Đọc ảnh
image = sitk.ReadImage("icon/Test_Image.dcm")

# Kiểm tra nếu ảnh không phải grayscale, chuyển đổi sang grayscale
if image.GetNumberOfComponentsPerPixel() > 1:
    image = sitk.VectorIndexSelectionCast(image, 0)

# Áp dụng BinaryThreshold
segmented_image = sitk.BinaryThreshold(image, lowerThreshold=100, upperThreshold=200, insideValue=1, outsideValue=0)

# Chuyển SimpleITK Image thành numpy array
segmented_array = sitk.GetArrayFromImage(segmented_image)

# Sử dụng squeeze để loại bỏ chiều có kích thước 1
segmented_array = segmented_array.squeeze()

# Hiển thị ảnh phân đoạn
plt.imshow(segmented_array, cmap="gray")
plt.show()
