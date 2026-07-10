import os
import psutil
import time
from ultralytics import YOLO

def get_gpu_temp():
    """Returns the GPU temperature in Celsius on Jetson using tegrastats."""
    try:
        output = os.popen("tegrastats | head -n 1").read()
        for item in output.split():
            if item.startswith("GPU@") and item.endswith("C"):
                temp_str = item.replace("GPU@", "").replace("C", "")
                return float(temp_str)
    except Exception as e:
        print(f"Error reading GPU temp: {e}")
    return None



def get_cpu_temps():
    """Returns a list of CPU core temperatures in Celsius on Jetson."""
    temps = []
    try:
        for zone in os.listdir("/sys/class/thermal"):
            if not zone.startswith("thermal_zone"):
                continue  # skip cooling_device or other entries

            path = f"/sys/class/thermal/{zone}/type"
            if os.path.exists(path):
                with open(path, "r") as f:
                    label = f.read().strip()
                if "cpu" in label.lower():
                    temp_path = f"/sys/class/thermal/{zone}/temp"
                    with open(temp_path, "r") as tf:
                        raw_temp = tf.read().strip()
                        temps.append(float(raw_temp) / 1000)
    except Exception as e:
        print(f"Error reading CPU temps: {e}")
    return temps

model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11n" if not provided
model_file = f"yolov{model_size}.pt"
try:
    model = YOLO(model_file)
except:
    model_file = f"yolo{model_size}.pt"
    model = YOLO(model_file) 
    
image_folder = r"coco128/images/train2017"  

cpu_usage_list = []
memory_usage_list = []
latency_list = []
gpu_temp_list = []
cpu_temp_list = []
num_images_processed = 0

process = psutil.Process(os.getpid())

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {image_name}...")
        
        start_time = time.time()

        results = model(image_path, save=False, show=False)
        latency_per_img = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']
        latency_list.append(latency_per_img)
        num_images_processed += 1  

        cpu_percent = process.cpu_percent(interval=None)
        memory_usage = process.memory_percent()

        cpu_usage_list.append(cpu_percent)
        memory_usage_list.append(memory_usage)

        # Collect temperatures
        gpu_temp = get_gpu_temp()
        if gpu_temp is not None:
            gpu_temp_list.append(gpu_temp)

        core_temps = get_cpu_temps()
        if core_temps:
            cpu_temp_list.extend(core_temps)

        time.sleep(0.1)  # Pause briefly to stabilize metric collection

# Compute averages and statistics
average_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
average_memory = sum(memory_usage_list) / len(memory_usage_list) if memory_usage_list else 0
latency = sum(latency_list) / num_images_processed if num_images_processed else 0
fps = 1000 / latency if latency else 0

# Temperature statistics
min_gpu_temp = min(gpu_temp_list) if gpu_temp_list else None
max_gpu_temp = max(gpu_temp_list) if gpu_temp_list else None
min_cpu_temp = min(cpu_temp_list) if cpu_temp_list else None
max_cpu_temp = max(cpu_temp_list) if cpu_temp_list else None

# Display results
print("\nInference completed for all images.")
print(f"Average CPU Usage (script-specific): {average_cpu:.2f}%")
print(f"Average Memory Usage (script-specific): {average_memory:.2f}%")
print(f"Average latency: {latency:.2f} ms")
print(f"FPS: {fps:.2f}")

# Print temperature stats
if min_gpu_temp is not None and max_gpu_temp is not None:
    print(f"Temp (min) GT: {min_gpu_temp:.2f}C")
    print(f"Temp (max) GT: {max_gpu_temp:.2f}C")

if min_cpu_temp is not None and max_cpu_temp is not None:
    print(f"Temp (min) Cores: {min_cpu_temp:.2f}C")
    print(f"Temp (max) Cores: {max_cpu_temp:.2f}C")
