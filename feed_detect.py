from ultralytics import YOLO
import cv2


model = YOLO("ultralytics/yolo/best(2).pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
results = model.predict(source="/home/gyorvaripeter/Letöltések/csgo_mirage_killfeed_frames/", show=False, imgsz=1280, save_txt=True, save=True) # Display preds. Accepts all YOLO predict arguments
