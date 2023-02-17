from ultralytics import YOLO
from argparse import ArgumentParser

# flagging operators
parser = ArgumentParser()
parser.add_argument(
    '--input', nargs='+', type=str, help='Path to sequence of videos or image folder, or just one video file.', default="./csgo_mirage_killfeed_frames"), 
parser.add_argument(
    '--model', type=str, help='Model name', default="./ultralytics/yolo/best_1280.pt")
parser.add_argument(
    '--imgsz', type=int, help='Detection image size', default=1280)
parser.add_argument(
    '--conf', type=float, help='Confidence value', default=0.25)

args = parser.parse_args()

model = YOLO(args.model)
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
test_video = "../Letöltések/csgo_test2.mp4"
test_images = "./csgo_mirage_killfeed_frames"
results = model.predict(source=args.input, show=False, imgsz=args.imgsz, conf=args.conf, save_txt=True, save=True, save_json=True) # Display preds. Accepts all YOLO predict arguments