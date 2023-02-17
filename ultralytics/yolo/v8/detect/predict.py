# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import json
import numpy as np

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=1, example=str(self.model.names), font=6)

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        
        return img

    def postprocess(self, preds, img, orig_img, classes=None):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
        return results

    def JSONWriter(self, frame, datalist):
        # a Python object (dict):
        x = {
        "FrameId%d" % frame: datalist
        }
        # convert into JSON:
        y = json.dumps(x)

        # the result is a JSON string:
        print(y, file= open(self.save_dir / 'Killrow.json','a')) 
    
    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        datalist = []
        rowindex_list = []
        sorted_datalist = []
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img or self.dataset.mode == 'image':  # batch_size >= 1
            #log_string += f'{idx}: '
            
            frame = self.dataset.count        
        else:
            frame = getattr(self.dataset, 'frame', 0)
            
        #frame = self.dataset.count
        
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        #log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
       # for c in det.cls.unique():
          #  n = (det.cls == c).sum()  # detections per class
           # log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
           ## killrow = (f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, ")

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            krow = d.xywh.tolist()
            rowindex=0
            if (krow[0][1] <= 95): rowindex=0
            elif(krow[0][1] > 95 and krow[0][1] <= 135): rowindex=1
            elif(krow[0][1] > 135 and krow[0][1] <= 175): rowindex=2
            elif(krow[0][1] > 175 and krow[0][1] <= 215): rowindex=3
            else:                                         rowindex=4
            rowindex_list.append(rowindex)
         #   print(krow[0][1],'\t', rowindex,'\t', self.model.names[int(cls)])    #bounding box y coordinate

            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
                
                datalist.append(self.model.names[c])
               # datalist.insert(rowindex,[self.model.names[c]])
            #    print(datalist)
               # print(self.model.names[c]) ########################################
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        sorted_datalist = sorted(map(list, zip(rowindex_list,datalist)))
        sorted_krow, _  = zip(*sorted_datalist)
       # print(sorted_datalist)
        # datalist sorting based on rowindex
        for i in range(len(sorted_datalist)):
            if (sorted_krow[i]==sorted_krow[i-1]):
                del sorted_datalist[i-1][0]
                sorted_datalist[i].extend(sorted_datalist[i-1])
                del sorted_datalist[i-1][0]
                # datalist[i].extend(datalist[i-1])
            else:
                del sorted_datalist[i-1][0]        

        sorted_datalist = [x for x in sorted_datalist if x != []]
        self.JSONWriter(frame,sorted_datalist)
        return log_string

def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or "yolov8s.pt"
    source = cfg.source if cfg.source is not None else ROOT / "assets" if (ROOT / "assets").exists() \
        else "https://ultralytics.com/images/bus.jpg"

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()

if __name__ == "__main__":
    predict()
