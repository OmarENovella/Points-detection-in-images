import cv2
import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def load_gray(path: str, size: tuple = (512, 512)):
    img = cv2.imread(path)
    resize_img = cv2.resize(img, size)
    return resize_img, cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

def find_dark_spots(gray, min_area=5, blur_ksize=3):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    spots = []
    for i in range(1, n_labels):
        size_px = stats[i, cv2.CC_STAT_AREA]
        if size_px < min_area:
            continue

        mask = (labels == i).astype(np.uint8) * 255
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = centroids[i]

        spots.append({
            "id": i,
            "centroid": (int(cx), int(cy)),
            "contour": contour[0],
            "size_px": int(size_px),
        })
    return spots

def draw_spots(img, spots, color=(255, 0, 255)):
    for s in spots:
        print(s)
        cv2.drawContours(img, [s["contour"]], -1, color, 1)
        cx, cy = s["centroid"]
        label = f'#{s["id"]} {s["size_px"]}px'
        cv2.putText(img, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img

if __name__ == "__main__":
    img, gray = load_gray("test/dispersed_dots.jpg")
    spots = find_dark_spots(gray)
    result = draw_spots(img, spots)

    cv2.imshow("Dark shapes detected", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()