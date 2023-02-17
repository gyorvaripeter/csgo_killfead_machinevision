# Program usage
Just simply run python3 feed_detect.py  (if you'd like to run detection on test images)
Otherwise you can set other input source with --input flag
e.g.: python3 feed_detect.py --input ~/test.mp4

# Log
The terminal shows you the processing time between frames in ms

# Dataset 
There are two main custom trained dataset. One of these trained 640 image size, and one of trained on 1280 image size. You can set the model with --model flag
e.g.: python3 feed_detect.py --input ~/test.mp4 --model best_1280.pt