import cv2

cap = cv2.VideoCapture(0)


def _apply_mask(img):
    def difference_of_gaussians(img, k1, s1, k2, s2):
        b1 = cv2.GaussianBlur(img, (k1, k1), s1)
        b2 = cv2.GaussianBlur(img, (k2, k2), s2)
        return b1 - b2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g_img = difference_of_gaussians(gray, 9, 9, 5, 11)
    th = cv2.threshold(g_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(c)
            extent = int(area) / (w * h)
            if extent > 0.6:
                return x, y, w, h

    return False, False, False, False


def extract_data(img, original_image):
    x, y, w, h = _apply_mask(img)
    if x:
        if h > w:
            x = x + int(h * 0.11)
            w = int(w * 0.1)
            y = y + int(h * 0.34)
            h = int(h * 0.17)
        else:
            x = x + int(w * 0.34)
            w = int(w * 0.17)
            y = y + int(h * 0.81)
            h = int(h * 0.08)
        roi = original_image[y: y + h, x: x + w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        original_image[y: y + roi.shape[0], x: x + roi.shape[1]] = roi
    return original_image


while True:
    ret, frame = cap.read()

    image = extract_data(frame, frame)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
