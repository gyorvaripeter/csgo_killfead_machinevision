from ultralytics import YOLO
import cv2


model = YOLO("ultralytics/yolo/best(2).pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
test_video = "../Letöltések/csgo_test2.mp4"
test_images = "./csgo_mirage_killfeed_frames"
results = model.predict(source=test_images, show=False, imgsz=1280, save_txt=True, save=True, save_json=True) # Display preds. Accepts all YOLO predict arguments