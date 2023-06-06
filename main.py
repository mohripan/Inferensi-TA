import numpy as np
import argparse
from Helpers.helpers import (
    get_image_from_sliding_window,
    forward_prop,
    forward_prop_without_slide,
    stitching_image,
    equalize_hist,
)
import cv2

def is_sliding(input_image, output_image, slide = True):
    if slide:
        sliding_images = get_image_from_sliding_window(input_image)
        gans_images = forward_prop(sliding_images,
                                   scale = 4,
                                   path = 'weights\RealESRGAN_x4.pth')
        stitched = stitching_image(gans_images)
        stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
        final_img = equalize_hist(stitched)
        cv2.imwrite(output_image, final_img)
        cv2.imshow("Stitched", final_img)
        cv2.waitKey(0)
        
    else:
        image = cv2.imread(input_image)
        image = forward_prop_without_slide(image,
                                           scale = 4,
                                           path = 'weights\RealESRGAN_x4.pth')

        image = np.array(image)
        image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
 
        clahe = cv2.createCLAHE(clipLimit = 10)
        final_img = clahe.apply(image_bw)
        cv2.imwrite(output_image, final_img)
        cv2.imshow("Result", final_img)
        cv2.waitKey(0)

def main(input_image, output_image) -> int:
    is_sliding(input_image, output_image, slide = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Super-resolution")
    parser.add_argument('-i', '--input', help='Input image path', default='Experiments/cc2.jpg')
    parser.add_argument('-o', '--output', help='Output image path', default='Experiments/output.png')
    args = parser.parse_args()

    main(args.input, args.output)