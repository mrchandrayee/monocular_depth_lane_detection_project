from ultralytics import YOLO
import cv2

VEHICLE_CLASSES = {"car","truck","bus","motorcycle","bicycle"}

class VehicleDetector:
    def __init__(self, weights="yolov8n.pt", conf=.35, device=None):
        self.model = YOLO(weights)
        self.conf, self.device = conf, device

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        detections = []
        names = result.names or {}
        if result.boxes is None: return detections
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            name = str(names.get(cls_id, cls_id)).lower()
            if name not in VEHICLE_CLASSES: continue
            x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().tolist())
            detections.append({
                "bbox":[x1,y1,x2,y2],
                "label":name,
                "confidence":float(box.conf[0].item())
            })
        return detections

    @staticmethod
    def draw(frame, detections):
        out=frame.copy()
        for d in detections:
            x1,y1,x2,y2=d["bbox"]
            cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
            label=f'{d["label"]} {d["confidence"]:.2f}'
            cv2.putText(out,label,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,
                        .5,(0,255,0),2,cv2.LINE_AA)
        return out
