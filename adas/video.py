import cv2

def open_writer(path,fps,width,height):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer
