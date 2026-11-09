import argparse
from pathlib import Path
import cv2
from adas.pipeline import ADASPipeline
from adas.video import open_writer

def main():
    p = argparse.ArgumentParser(description="Dashcam lane, vehicle and monocular depth visualization")
    p.add_argument("--source", required=True, help="Video/image path or webcam index")
    p.add_argument("--output", default="runs/adas")
    p.add_argument("--yolo", default="yolov8n.pt", help="Ultralytics YOLO weights")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default=None, help="cpu, 0, etc.")
    p.add_argument("--depth", action="store_true", help="Enable MiDaS relative depth (requires optional dependencies)")
    p.add_argument("--depth-model", default="DPT_Hybrid", choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-yolo", action="store_true", help="Run lane detection only")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pipeline = ADASPipeline(yolo_weights=None if args.no_yolo else args.yolo,
                            conf=args.conf, device=args.device,
                            enable_depth=args.depth, depth_model=args.depth_model)
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if isinstance(source, str) and Path(source).suffix.lower() in image_ext:
        frame = cv2.imread(source)
        if frame is None: raise FileNotFoundError(source)
        result = pipeline.process(frame)
        cv2.imwrite(str(out / "adas_dashboard.jpg"), result)
        if args.show:
            cv2.imshow("ADAS", result); cv2.waitKey(0); cv2.destroyAllWindows()
        print("Saved:", out / "adas_dashboard.jpg")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps < 1 or fps > 120: fps = 25.0
    writer = None; frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            dashboard = pipeline.process(frame)
            if writer is None:
                h,w = dashboard.shape[:2]
                writer = open_writer(out / "adas_output.mp4", fps, w, h)
            writer.write(dashboard)
            if args.show:
                cv2.imshow("ADAS - press q to quit", dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            frames += 1
    finally:
        cap.release()
        if writer: writer.release()
        if args.show: cv2.destroyAllWindows()
    print(f"Processed {frames} frames. Output: {out}")

if __name__ == "__main__":
    main()
