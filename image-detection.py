import re
import cv2
import pytesseract
import numpy as np
from pytesseract import Output

IMAGE_PATH = ''


def _get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _remove_noise(image, kernel=(1, 1)):
    return cv2.GaussianBlur(image, kernel, 0)


def _thresholding(image):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 51
    )


def _apply_mask(img):
    def difference_of_gaussians(img, k1, s1, k2, s2):
        b1 = cv2.GaussianBlur(img, (k1, k1), s1)
        b2 = cv2.GaussianBlur(img, (k2, k2), s2)
        return b1 - b2

    img2 = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    g_img = difference_of_gaussians(gray, 9, 9, 3, 11)
    th = cv2.threshold(g_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    img1 = img2.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(c)
            extent = int(area) / (w * h)
            if extent > 0.6:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                mask = np.zeros_like(img1)
                mask = cv2.rectangle(
                    mask, (x, y), (x + w, y + h), (255, 255, 255), -1
                )
                return cv2.bitwise_and(img1, mask)


def _bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def _sharp(image):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(image, -1, kernel)


def _contrast(image):
    alpha = 1.5
    beta = 10
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def preprocess_image(img, kernel=1, mask=False):
    if mask:
        img = _apply_mask(img)
        img = _bilateral_filter(img)
        img = _sharp(img)
        img = _contrast(img)

    img = _get_grayscale(img)
    img = _remove_noise(img, (kernel, kernel))
    img = _thresholding(img)
    return img


def _get_data_position(img):
    x, y, w, h = False, False, False, False
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    number_pattern = '[0-9]{11}'

    n_boxes = len(d['text'])
    print(d['text'])
    for i in range(n_boxes):
        if re.match(number_pattern, d['text'][i]):
            (x, y, w, h) = (
                d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            )
    return (x, y, w, h)


def extract_data(img, original_image):
    x, y, w, h = _get_data_position(img)
    if x:
        roi = original_image[y: y + h, x: x + w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        original_image[y: y + roi.shape[0], x: x + roi.shape[1]] = roi
    return original_image, x


image = cv2.imread(IMAGE_PATH)
original_image = image.copy()
kernel = 1
success = False
mask = False

while not success and kernel <= 9 and not (kernel == 11 and mask):
    img = original_image.copy()
    img = preprocess_image(img, kernel, mask)
    image, success = extract_data(img, image)

    kernel += 2
    if kernel == 11 and not mask:
        kernel = 1
        mask = True


cv2.imshow('DNI Original', original_image)
cv2.imshow('DNI Blur', image)
cv2.waitKey(0)
