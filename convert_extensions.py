import os
from PIL import Image

def convert_images_to_jpeg(input_folder, output_folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension in image_extensions:
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Convert image to RGB mode
            image = image.convert('RGB')
            
            # Save the image in JPEG format
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path, "JPEG")

input_folder = "Data/HighResolution"
output_folder = "Dataset/HighResolution"

convert_images_to_jpeg(input_folder, output_folder)

input_folder = "Data/LowResolution"
output_folder = "Dataset/LowResolution"

convert_images_to_jpeg(input_folder, output_folder)