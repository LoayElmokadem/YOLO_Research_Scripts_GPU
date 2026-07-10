import os
import time
from ultralytics import YOLO

model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Set via YOLO_MODEL_SIZE env var
model_file = f"yolov{model_size}.pt"

try:
    start_time = time.time()
    model = YOLO(model_file)
except Exception:
    model_file = f"yolo{model_size}.pt"
    start_time = time.time()
    model = YOLO(model_file)

end_time = time.time()
load_time = end_time - start_time

print(f"Model load time: {load_time:.3f} seconds")
