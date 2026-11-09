import cv2
import numpy as np

class LaneDetector:
    """Classical lane-marking detector using ROI, Canny, Hough and line fitting."""
    def __init__(self):
        self.previous = None

    def detect(self, frame):
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 55, 150)
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(.05*w),h), (int(.42*w),int(.56*h)),
            (int(.58*w),int(.56*h)), (int(.98*w),h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 35,
                                minLineLength=max(25,int(w*.08)),
                                maxLineGap=max(15,int(w*.04)))
        left, right = [], []
        if lines is not None:
            for line in lines[:,0]:
                x1,y1,x2,y2 = map(int,line)
                dx = x2-x1
                if dx == 0: continue
                slope = (y2-y1)/dx
                if abs(slope) < .35 or abs(slope) > 3.5: continue
                intercept = y1 - slope*x1
                (left if slope < 0 else right).append((slope,intercept))
        result = []
        for group in (left,right):
            if not group: continue
            slope, intercept = np.mean(group,axis=0)
            y_bottom, y_top = h, int(.60*h)
            x_bottom = int((y_bottom-intercept)/slope)
            x_top = int((y_top-intercept)/slope)
            result.append(((x_bottom,y_bottom),(x_top,y_top)))
        return result

    @staticmethod
    def draw(frame, lines):
        overlay = frame.copy()
        for p1,p2 in lines:
            cv2.line(overlay,p1,p2,(0,255,255),max(2,frame.shape[1]//400),cv2.LINE_AA)
        return cv2.addWeighted(frame,.72,overlay,.8,0)
