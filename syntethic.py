import os
import cv2
import glob

input_dir = "Dataset/HighResolution"
output_dir = "Dataset/LowResolution"
interpolation_methods = {
    "bicubic": cv2.INTER_CUBIC,
    "nearest_neighbor": cv2.INTER_NEAREST,
    "lanczos": cv2.INTER_LANCZOS4
}

# Create output directories for each interpolation method
os.makedirs(output_dir, exist_ok=True)
for method in interpolation_methods.keys():
    os.makedirs(os.path.join(output_dir, method), exist_ok=True)

# Load high-resolution images
high_res_images = glob.glob(os.path.join(input_dir, "*.jpg"))

for img_path in high_res_images:
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)

    # Downscale the image by a factor of 4 (adjust this factor as needed)
    target_width, target_height = img.shape[1] // 4, img.shape[0] // 4

    for method, interpolation in interpolation_methods.items():
        # Downscale the image using the current interpolation method
        low_res_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)

        # Save the low-resolution image
        output_path = os.path.join(output_dir, method, img_name)
        cv2.imwrite(output_path, low_res_img)
        print(f"Saved {output_path}")

print("All synthetic low-resolution images generated.")