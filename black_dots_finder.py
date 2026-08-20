import cv2
import numpy as np

def load_gray_image(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def calculate_blackhat(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detect_and_group_dark_zones(blackhat, thresh_val=30, close_kernel_size=(15, 15), min_area=1):
    _, binary = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2
        zones.append({
            "area": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "center": (cx, cy)
        })
    return binary, grouped, zones

def mark_zones(image, zones):
    result = image.copy()
    for idx, zone in enumerate(zones, start=1):
        x = zone["x"]
        y = zone["y"]
        w = zone["width"]
        h = zone["height"]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(result, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return result

if __name__ == "__main__":
    FILE_PATH = "test/real_image_test.jpg"
    img, gray = load_gray_image(FILE_PATH)
    blackhat = calculate_blackhat(gray)
    binary, grouped, zones = detect_and_group_dark_zones(blackhat)
    print(zones)
    result = mark_zones(img, zones)
    cv2.imshow("Original", img)
    cv2.imshow("Black Hat", blackhat)
    #cv2.imshow("Binary", binary)
    #cv2.imshow("Grouped", grouped)
    cv2.imshow("Detected Zones", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()