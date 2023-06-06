import os
from PIL import Image
from shutil import copyfile

def get_image_resolution(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height

def evaluate_and_copy_high_resolution_images(input_folder, high_res, threshold_width=512, threshold_height=512):
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    if not os.path.exists(high_res):
        os.makedirs(high_res)

    for filename in os.listdir(input_folder):
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension in image_extensions:
            image_path = os.path.join(input_folder, filename)
            width, height = get_image_resolution(image_path)

            if width >= threshold_width and height >= threshold_height:
                print(f"The image {filename} has a high resolution: {width}x{height}")
                output_path = os.path.join(high_res, filename)
                copyfile(image_path, output_path)
            else:
                print(f"The image {filename} has a low resolution: {width}x{height}")
                
input_folder = "Dataset"
high_res = "Data/HighResolution"

evaluate_and_copy_high_resolution_images(input_folder, high_res)