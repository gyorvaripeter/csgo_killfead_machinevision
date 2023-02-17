# Program usage
Just simply run python3 feed_detect.py  (if you'd like to run detection on test images)

Otherwise you can set other parameters with argument flags:
-   other input source with --input FILEPATH
e.g.: python3 feed_detect.py --input ~/test.mp4
-   other image size --imgsz INT
-   other model --model FILEPATH
-   confidence value --conf FLOAT

# Log
-   The terminal shows you the processing time between frames in ms
-   The JSON file store the framenumber and the detected weapons and actions

# Dataset 
There are two main custom trained dataset. One of these trained 640 image size, and one of trained on 1280 image size. You can set the model with --model flag (640 models wasn't good enough so please try with 1280)

e.g.: python3 feed_detect.py --input ~/test.mp4 --model best_1280.pt  <-- that's the default