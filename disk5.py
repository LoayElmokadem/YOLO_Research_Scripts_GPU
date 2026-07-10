import psutil
import time
import os
from PIL import Image
from ultralytics import YOLO

model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolov{model_size}.pt"
try:
    model = YOLO(model_file)  # Repla
except:
    model_file = f"yolo{model_size}.pt"
    model = YOLO(model_file)
    

image_directory = r"coco128/images/train2017"

image_paths = [
    os.path.join(image_directory, img) 
    for img in os.listdir(image_directory) 
    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
]

images = {path: Image.open(path) for path in image_paths}

total_elapsed_time = 0
total_read_bytes = 0
total_write_bytes = 0
total_images = len(image_paths)

if total_images == 0:
    print("No images found in the directory.")
else:
    process = psutil.Process(os.getpid())

    for image_path, image in images.items():
        io_start = process.io_counters()

        start_time = time.time()
        results = model(image, save=True, show=False)
        end_time = time.time()

        io_end = process.io_counters()

        elapsed_time = end_time - start_time
        read_bytes = (io_end.read_bytes - io_start.read_bytes) / (1024**2)  # MB
        write_bytes = (io_end.write_bytes - io_start.write_bytes) / (1024**2)  # MB

        total_elapsed_time += elapsed_time
        total_read_bytes += read_bytes
        total_write_bytes += write_bytes

        print(f"Processed {os.path.basename(image_path)}: {elapsed_time:.2f}s, {read_bytes:.2f}MB read, {write_bytes:.2f}MB written")

    avg_elapsed_time = total_elapsed_time / total_images
    avg_read_bytes = total_read_bytes / total_images
    avg_write_bytes = total_write_bytes / total_images

    avg_read_speed = avg_read_bytes / avg_elapsed_time if avg_elapsed_time > 0 else 0
    avg_write_speed = avg_write_bytes / avg_elapsed_time if avg_elapsed_time > 0 else 0

    print("\nSummary for all images:")
    print(f"Total Images: {total_images}")
    print(f"Average Elapsed Time: {avg_elapsed_time:.4f} seconds")
    print(f"Average Read Bytes: {avg_read_bytes:.4f} MB")
    print(f"Average Write Bytes: {avg_write_bytes:.4f} MB")
    print(f"Average Disk Read Speed: {avg_read_speed:.4f} MB/s")
    print(f"Average Disk Write Speed: {avg_write_speed:.4f} MB/s")
