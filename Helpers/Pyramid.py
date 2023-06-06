import os
import sys
import numpy as np
import cv2
import argparse
from PIL import Image
from helpers import pyramid, pyramid_gaussian

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap.add_argument("-s", "--scale", type = float, default = 1.5, help = "scale factor size")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    
    for (i, resized) in enumerate(pyramid(image, scale = args["scale"])):
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)


main()