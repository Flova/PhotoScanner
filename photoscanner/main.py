#!/usr/bin/env python3

import os
import typing
import time
import cv2
import imutils
import numpy as np
import argparse
from pathlib import Path

params: typing.Dict[str, object] = dict()

def process_image(image: np.ndarray) -> np.ndarray:
    ratio = image.shape[0] / 500.0
    margin = 50

    img_small = imutils.resize(image, height = 500)

    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    mask = 255 - cv2.inRange(
        cv2.medianBlur(
            img_hsv.copy(), params['median']),
            tuple([x for x in params["min_hsv"].split(",")]),
            tuple([x for x in params["max_hsv"].split(",")]))

    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt: np.ndarray = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.085 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    rect *= ratio

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    crop_img = warp[margin:-margin, margin:-margin]

    return crop_img


def fix_aspect_ratio(image: np.ndarray, wanted_width: float, wanted_height: float) -> np.ndarray:
    width = image.shape[1]
    height = int(image.shape[1] * (wanted_height / wanted_width))
    dim = (width, height)
    # resize image
    fixed = cv2.resize(image, dim)
    return fixed


def run_for_folder(input_directory: str, output_directory: str, size: typing.Tuple[float, float]) -> None:
    directory = input_directory
    #size = (14.9, 10.1)

    print(input_directory, output_directory, size)
    return

    try:
        os.mkdir(output_directory)
    except OSError:
        print ("Creation of the directory %s failed" % output_directory)

    cv2.imshow("Image", np.zeros((50,50)))

    print(f"Number of items {len(os.listdir(directory))}!")

    # all files in folder
    for file in os.listdir(directory):
        if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".png"):
            print(f"Execute for file {file}!")
            # Load
            image = cv2.imread(os.path.join(directory, file))
            # Exec
            warped_image = process_image(image)

            try:
                # Fix aspect
                warped_image = fix_aspect_ratio(warped_image, *size)
                # Write
                cv2.imwrite(os.path.join(output_directory, file), warped_image)
                # Viewer
                img_warp_small = imutils.resize(warped_image.copy(), height = 500)
                cv2.imshow("Image", img_warp_small)
                cv2.waitKey(1)
            except:
                print(f"error at image {file}")
    cv2.destroyAllWindows()


def run():
    parser = argparse.ArgumentParser(
        description='A simple tool to digitalize printed photos using a greenscreen and a DSLR.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input-folder', default=".", type=str, help='Input folder with the images captured by the DSLR.')
    parser.add_argument('-o', '--output-folder', default=None, type=str, help='Folder in which the croped images are saved.')
    parser.add_argument('-he', '--height', type=float, default=8.8, help='Height of the input images in cm (both hight and width are required).')
    parser.add_argument('-w', '--width', type=float, default=13, help='Width of the input images in cm (both hight and width are required).')
    parser.add_argument('-m', '--median', type=int, default=21, help='Median filter kernel size.')
    parser.add_argument('--max-hsv', type=str, default="50,130,40", help='Max HSV greenscreen thresh')
    parser.add_argument('--min-hsv', type=str, default="80,255,250", help='Min HSV greenscreen thresh')

    args = parser.parse_args()

    params = vars(args)

    if args.output_folder is None:
        args.output_folder = os.path.join(str(Path(args.input_folder).resolve()), "converted")

    run_for_folder(
        str(Path(args.input_folder).resolve()),
        str(Path(args.output_folder).resolve()),
        (args.width, args.height))


if __name__ == "__main__":
    run()
