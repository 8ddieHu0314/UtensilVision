import cv2

class VideoWriter:
    def __init__(self, filename: str, fps: float = 20.0, frame_size: tuple = (640, 480)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.filename = filename

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def get_filename(self):
        return self.filename