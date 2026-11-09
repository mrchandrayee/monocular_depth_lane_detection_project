import cv2
import numpy as np

class MiDaSDepth:
    """Relative monocular depth. Values are normalized for visualization, not metres."""
    def __init__(self, model_type="DPT_Hybrid"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Depth requires torch and torchvision. Install requirements-depth.txt") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform

    def predict(self, frame):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with self.torch.no_grad():
            prediction = self.model(inp)
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        lo,hi=np.percentile(depth,[2,98])
        depth=np.clip((depth-lo)/(hi-lo+1e-6),0,1)
        # MiDaS output is inverse-depth-like: brighter generally means nearer.
        return (depth*255).astype(np.uint8)
