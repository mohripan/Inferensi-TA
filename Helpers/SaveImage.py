from helpers import get_image_from_sliding_window
import argparse
import cv2
from PIL import Image

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'Path to the image')
    args = vars(ap.parse_args())
    
    images = get_image_from_sliding_window(args['image'])
    
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img.save(f'Helpers/ImageTest/hasil{str(i)}.jpg')
    
main()