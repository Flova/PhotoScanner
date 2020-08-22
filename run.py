import os
import time
import cv2
import imutils
import numpy as np


def process_image(image):
    ratio = image.shape[0] / 500.0
    margin = 50

    img_small = imutils.resize(image, height = 500)

    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    mask = 255 - cv2.inRange(cv2.medianBlur(img_hsv.copy(), 21), (50,130,40), (80,255,250))

    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

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


def fix_aspect_ratio(image, wanted_width, wanted_height):


    width = image.shape[1]
    height = int(image.shape[1] * (wanted_height / wanted_width))
    dim = (width, height)
    # resize image
    fixed = cv2.resize(image, dim)
    return fixed

directory = "/run/media/florian/C9E9-27B2/DCIM/136MSDCF/"
size = (13, 8.8)
#size = (14.9, 10.1)

output_directory = os.path.join(directory, "warped/")

try:
    os.mkdir(output_directory)
except OSError:
    print ("Creation of the directory %s failed" % output_directory)

cv2.imshow("Image", np.zeros((50,50)))

print(f"So many items {len(os.listdir(directory))}!")

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
