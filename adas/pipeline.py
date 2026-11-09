import cv2
import numpy as np
from .lanes import LaneDetector
from .objects import VehicleDetector

class ADASPipeline:
    def __init__(self, yolo_weights="yolov8n.pt", conf=.35, device=None,
                 enable_depth=False, depth_model="DPT_Hybrid"):
        self.lanes=LaneDetector()
        self.vehicles=VehicleDetector(yolo_weights,conf,device) if yolo_weights else None
        self.depth=None
        if enable_depth:
            from .depth import MiDaSDepth
            self.depth=MiDaSDepth(depth_model)

    def process(self, frame):
        road=self.lanes.draw(frame,self.lanes.detect(frame))
        detections=self.vehicles.detect(frame) if self.vehicles else []
        road=VehicleDetector.draw(road,detections) if self.vehicles else road
        h,w=frame.shape[:2]
        # Right-side panel is a lightweight schematic scene/depth view.
        panel_w=max(240,int(w*.48))
        panel_h=h
        if self.depth:
            depth=self.depth.predict(frame)
            depth_color=cv2.applyColorMap(depth,cv2.COLORMAP_INFERNO)
            panel=cv2.resize(depth_color,(panel_w,panel_h))
            cv2.putText(panel,"RELATIVE DEPTH",(12,28),cv2.FONT_HERSHEY_SIMPLEX,
                        .65,(255,255,255),2,cv2.LINE_AA)
        else:
            panel=np.full((panel_h,panel_w,3),24,np.uint8)
            self._draw_depth_grid(panel)
        # Draw approximate projected object cuboids in panel using image boxes.
        for d in detections:
            x1,y1,x2,y2=d["bbox"]
            cx=(x1+x2)/2/w
            cy=(y1+y2)/2/h
            bw=(x2-x1)/w
            # Relative scale proxy: nearer objects appear larger/lower.
            px=int(cx*panel_w)
            py=int(cy*panel_h)
            box_w=max(18,int(bw*panel_w*.65))
            box_h=max(22,int((y2-y1)/h*panel_h*.55))
            self._draw_wire_box(panel,px,py,box_w,box_h)
        # Side-by-side display: road view + depth/schematic panel.
        road_w=int(panel_h*frame.shape[1]/frame.shape[0])
        road=cv2.resize(road,(road_w,panel_h))
        return np.hstack([road,panel])

    @staticmethod
    def _draw_depth_grid(panel):
        h,w=panel.shape[:2]
        center=w//2
        for y in range(int(h*.28),h, max(24,h//12)):
            cv2.line(panel,(0,y),(w,y),(65,65,65),1)
        for x in range(-w,2*w,max(35,w//8)):
            cv2.line(panel,(center,int(h*.28)),(x,h),(55,55,55),1)
        cv2.putText(panel,"DEPTH SCHEMATIC",(12,28),cv2.FONT_HERSHEY_SIMPLEX,
                    .6,(0,255,80),2,cv2.LINE_AA)
        cv2.putText(panel,"Enable --depth for MiDaS",(12,h-18),
                    cv2.FONT_HERSHEY_SIMPLEX,.43,(180,180,180),1,cv2.LINE_AA)

    @staticmethod
    def _draw_wire_box(panel,cx,cy,bw,bh):
        front=[(cx-bw//2,cy-bh//2),(cx+bw//2,cy-bh//2),
               (cx+bw//2,cy+bh//2),(cx-bw//2,cy+bh//2)]
        dx=max(8,bw//3); dy=max(8,bh//4)
        back=[(x+dx,y-dy) for x,y in front]
        for i in range(4):
            cv2.line(panel,front[i],front[(i+1)%4],(80,220,80),2)
            cv2.line(panel,back[i],back[(i+1)%4],(80,220,80),1)
            cv2.line(panel,front[i],back[i],(80,220,80),1)
