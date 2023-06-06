import imutils
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ESRGAN import RealESRGAN
from PIL import Image
from skimage.transform import pyramid_gaussian
torch.backends.cudnn.benchmark = True


def pyramid(image, scale = 1.5, minSize = (30, 30)):
    yield image
    
    i = 0
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
        
def pyramid_gaussian(image, downscale = 2):
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale = 2)):
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)
        
def split_image_with_overlap(image_path, num_slices = 6, overlap=0.10, output_folder = 'Output'):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    cell_h, cell_w = h // num_slices, w // num_slices

    # Calculate the step size, accounting for overlap
    step_h = int(cell_h * (1 - overlap))
    step_w = int(cell_w * (1 - overlap))

    slices = []
    
    idx = 0

    for y in range(0, h - cell_h + 1, step_h):
        for x in range(0, w - cell_w + 1, step_w):
            cell = image[y:y+cell_h, x:x+cell_w]
            cv2.imwrite(os.path.join(output_folder, f'slice_{idx}.png'), slice)
            idx += 1
            slices.append(cell)

    return slices
        
def sliding_window(image, step_size, window_size):
    if image.shape[1] > image.shape[0]:
        for x in range(0, image.shape[1], step_size):
            yield (x, image[0:, x:x + window_size[0]])
            
    else:
        for y in range(0, image.shape[0], step_size):
            yield(y, image[y:y + window_size[1], 0:])
            
def sliding_window_with_overlap(image, step_size, window_size, overlap=0.15):
    # Calculate the overlap in pixels
    overlap_pixels_w = int(window_size[0] * overlap)
    overlap_pixels_h = int(window_size[1] * overlap)

    # Slide the window over the image
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size - overlap_pixels_h):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size - overlap_pixels_w):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
            
def get_image_from_sliding_window(image_path):
    image = cv2.imread(image_path)
    images = []
    winH, winW = (600, 600)
    
    for (x, window) in sliding_window(image, step_size = 32, window_size = (winW, winH)):
        if image.shape[1] > image.shape[0]:
            if window.shape[1] != winW:
                continue
            
            clone = image.copy()
            clone = clone[0:, x:x + winW, :]
            images.append(clone)

        else:
            if window.shape[0] != winH:
                continue
            
            clone = image.copy()
            clone = clone[x:x + winH, 0:]
            images.append(clone)
        
    return images

def forward_prop(image, scale = 2, path = 'weights/RealESRGAN_x2.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale = scale)
    model.load_weights(path, download = True)
    images = []
    
    for i, img in enumerate(image):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f'berhasil {i+1}')
        sr_image = model.predict(img)
        images.append(sr_image)
        
    return images

def forward_prop_without_slide(image, scale = 2, path = 'weights/RealESRGAN_x2.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale = scale)
    model.load_weights(path, download = True)
    sr_image = model.predict(image)
    return sr_image

def stitching_image(read_images, crop = 1):
    print("[INFO] loading images...")
    images = []
    for image in read_images:
        image = np.array(image)
        images.append(image)
        
    print('[INFO] stitching images...')
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    print(status)
    
    if status == 0:
        if crop == 0:
            print('[INFO] cropping...')
            stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                          cv2.BORDER_CONSTANT, (0, 0, 0))
            
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grad_contours(cnts)
            c = max(cnts, key = cv2.contourArea)
            
            mask = np.zeros(thresh.shape, dtype = 'uint8')
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            minRect = mask.copy()
            sub = mask.copy()
            
            while cv2.countNonZero(sub) > 0:
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)
                
            cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key = cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            
            stitched = stitched[y:y + h, x:x + w]
    else:
        print('[INFO] image stitching failed ({})'.format(status))
        
    return stitched

def concatenate_images(images, num_cols=3):
    # Convert the list of images to a list of numpy arrays
    images = [np.array(image) for image in images]

    # Reshape the list of images into a 2D list
    images_2d = [images[i:i+num_cols] for i in range(0, len(images), num_cols)]

    # Concatenate the images
    concatenated_images = [np.concatenate(row_images, axis=1) for row_images in images_2d]  # Concatenate each row
    concatenated_image = np.concatenate(concatenated_images, axis=0)  # Concatenate all rows

    return concatenated_image

def equalize_hist(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(image_bw)
    
    return final_img

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Create a CLAHE object with the specified parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the image
    result = clahe.apply(image)
    
    return result