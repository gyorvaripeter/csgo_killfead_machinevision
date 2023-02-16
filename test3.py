import cv2
import numpy as np

# Load the YOLOv3 weights and configuration file
net = cv2.dnn.readNet("/ultralytics/yolo/best(1).pt", "yolov3.cfg")

# Load the custom class names and colors
classes = ["class1", "class2", "class3"]
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

# Set the confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Load the image and crop to the row of interest
img = cv2.imread("image.jpg")
height, width, _ = img.shape
row_start = 100  # Starting pixel of the row of interest
row_end = 200    # Ending pixel of the row of interest
img_row = img[row_start:row_end, 0:width]

# Create a blob from the cropped image and pass it through the network
blob = cv2.dnn.blobFromImage(img_row, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())

# Loop over the outputs and detect objects
class_ids = []
confidences = []
boxes = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * (row_end-row_start) + row_start)
            w = int(detection[2] * width)
            h = int(detection[3] * (row_end-row_start))
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Loop over the indices and draw the boxes
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    class_id = class_ids[i]
    color = colors[class_id]
    cv2.rectangle(img, (x, y+row_start, w, h), color, 2)
    cv2.putText(img, classes[class_id], (x, y+row_start-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with the detected objects
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
