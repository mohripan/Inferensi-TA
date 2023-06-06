from helpers import sliding_window
import argparse
import time
import cv2

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'Path to the image')
    args = vars(ap.parse_args())
    
    image = cv2.imread(args['image'])
    (winW, winH) = (image.shape[1] // 2, image.shape[0] // 2)
    
    for (x, window) in sliding_window(image, step_size = 32, window_size = (winW, winH)):
        if window.shape[1] != winW:
            continue
        
        clone = image.copy()
        clone = clone[0:, x:x + winW, :]
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow('Window', clone)
        cv2.waitKey(1)
        time.sleep(0.025)
    
    # for resized in pyramid(image, scale = 1.5):
    #     for (x, y, window) in sliding_window(resized, step_size = 32, window_size = (winW, winH)):
    #         if window.shape[0] != winH or window.shape[1] != winW:
    #             continue
            
    #         clone = resized.copy()
    #         cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    #         cv2.imshow('Window', clone)
    #         cv2.waitKey(1)
    #         time.sleep(0.025)
main()