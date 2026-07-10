import os 
from ultralytics import YOLO
import torch
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  
model_file = f"yolov{model_size}.pt"

try:
    model = YOLO(model_file)
except:
    model_file = f"yolo{model_size}.pt"
    model = YOLO(model_file)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


print(f"Running validation on: {model_file}")

results = model.val(
    data="coco128.yaml",
    batch=1,         # Keep it 1 to avoid OOM crashes
    imgsz=320,       # Lower resolution to fit GPU memory (you can try 416 if needed)
    workers=0,       # Multiprocessing causes issues on Nano, keep it 0
    max_det=20,      # Reasonable limit, reduces postprocessing time
    plots=False      # Avoids saving images that consume memory/disk
)

print("Validation complete.")