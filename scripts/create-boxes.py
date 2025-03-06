import cv2
import json

boxes = []
start = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global start, drawing, boxes, img

    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = img.copy()
        cv2.rectangle(temp, start, (x, y), (0, 255, 0), 2)
        cv2.imshow("image", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        end = (x, y)
        drawing = False
        x0, y0 = min(start[0], end[0]), min(start[1], end[1])
        x1, y1 = max(start[0], end[0]), max(start[1], end[1])
        boxes.append((x0, y0, x1, y1))
        print("New box:", boxes[-1])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow("image", img)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python create-boxes.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}!")
        sys.exit(1)

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("image", 800, 600)
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("All boxes:", boxes)

    # Always write temp_boxes.json (even if empty)
    with open("temp_boxes.json", "w") as f:
        json.dump(boxes, f)
    print("Saved bounding boxes to temp_boxes.json")
