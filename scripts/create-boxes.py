import cv2

boxes = []
start = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global start, drawing, boxes, img

    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Draw a preview rectangle on a copy of the image
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
        # Draw the finalized rectangle on the main image
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow("image", img)

# Load your image
img = cv2.imread("../data/refined/img-stones/FNE_24_778.jpg")
if img is None:
    print("Error: Could not load image!")
    exit()

# Create a resizable window (often helps on macOS)
cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("image", 800, 600)
cv2.setMouseCallback("image", draw_rectangle)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    # Press ESC or 'q' to quit
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
print("All boxes:", boxes)
