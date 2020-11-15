import numpy as np
import cv2
import pytesseract

filename = ""
threshold = False
deNoise = False
skew = False
image = cv2.imread(filename, 0)

if threshold:
    image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]

if deNoise:
    image = cv2.fastNlMeansDenoising(image, 12, 12, 7, 21)

if skew:
    image = cv2.bitwise_not(image)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(cords)[-1]
    h = 0
    w = 0
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    image = cv2.bitwise_not(image)

text = pytesseract.image_to_string(image)
with open('file.txt', 'w') as file:
    file.write(text)
