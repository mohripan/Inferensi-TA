import os
from PIL import Image

def create_low_resolution_images(input_folder, output_folder, scale_factor=0.5):
    image_extensions = ['.jpg', '.jpeg', '.png']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension in image_extensions:
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            width, height = image.size

            # Calculate the dimensions of the low-resolution image
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            low_res_image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # Save the low-resolution image
            output_path = os.path.join(output_folder, filename)
            low_res_image.save(output_path)

input_folder = "Data/HighResolution"
output_folder = "Data/LowResolution"

create_low_resolution_images(input_folder, output_folder)