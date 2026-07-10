# Docker Setup — NVIDIA Jetson Nano & Jetson Orin Nano

This guide covers running the YOLO benchmark suite inside a Docker container on **NVIDIA Jetson devices**. Each device uses a different image — see the section for your hardware below.

> **Choose your device section below** — the Jetson Nano (original) and Jetson Orin Nano use different JetPack versions and Docker image tags.

---

## 🟢 Jetson Orin Nano (JetPack 5 / JetPack 6)

We use the official **Ultralytics Jetson images** — purpose-built for YOLO, actively maintained, and pre-loaded with CUDA-enabled PyTorch.

> **Tip:** Check your JetPack version first:
> ```bash
> cat /etc/nv_tegra_release
> ```

### Pull the Image

```bash
# JetPack 5 (Ubuntu 20.04 — L4T R35.x)
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack5

# JetPack 6 (Ubuntu 22.04 — L4T R36.x)
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack6
```

### Run Command

```bash
sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    --ipc host \
    --privileged \
    -v ~/YOLO_RESEARCH_GPU_REPO:/workspace \
    -v /usr/bin/tegrastats:/usr/bin/tegrastats \
    -w /workspace \
    ultralytics/ultralytics:latest-jetson-jetpack5 \
    /bin/bash
```

> **Note:** Swap `jetpack5` → `jetpack6` to match your JetPack version. Replace `~/YOLO_RESEARCH_GPU_REPO` with your actual repo path.

### Argument Reference

| Argument | Purpose |
|----------|---------|
| `-it` | Interactive terminal |
| `--rm` | Remove container when it exits |
| `--runtime nvidia` | Expose the Jetson GPU to the container |
| `--network host` | Share the host's network (useful for cameras / ROS) |
| `--ipc host` | Share IPC memory (improves PyTorch/DataLoader performance) |
| `--privileged` | Grants access to USB devices, cameras, GPIO, etc. |
| `-v ~/YOLO_RESEARCH_GPU_REPO:/workspace` | Mount the project directory into the container |
| `-v /usr/bin/tegrastats:/usr/bin/tegrastats` | Expose `tegrastats` for thermal monitoring |
| `-w /workspace` | Start the shell in the project directory |

---

## 🔵 Jetson Nano — Original (JetPack 4)

### Pull the Image

```bash
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
```

### Full Run Command

```bash
sudo docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --privileged \
  -v "/home/<your_user>/YOLO_RESEARCH_GPU_REPO":/app \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY="$DISPLAY" \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats \
  -w /app \
  ultralytics/ultralytics:latest-jetson-jetpack4 \
  /bin/bash
```

---

## 🔧 Install Missing Dependencies Inside the Container

Run these inside the container after it starts:

```bash
apt update && apt install -y libgl1-mesa-glx
pip install ultralytics opencv-python psutil gpustat
```

---

## ✅ Quick Smoke Test

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo12n.pt')
results = model('https://ultralytics.com/images/zidane.jpg')
for r in results:
    print(f'Detections: {len(r.boxes)}')
"
```

---

## 🚀 Run the Full Benchmark Suite

```bash
bash all_scripts2.sh
```

See [README.md](README.md) for detailed instructions on configuring model sizes and interpreting results.

---

## 📌 Image Reference

| Device | JetPack | Docker Image |
|--------|---------|-------------|
| Jetson Orin Nano | JetPack 6 (Ubuntu 22.04) | `ultralytics/ultralytics:latest-jetson-jetpack6` |
| Jetson Orin Nano | JetPack 5 (Ubuntu 20.04) | `ultralytics/ultralytics:latest-jetson-jetpack5` |
| Jetson Nano (original) | JetPack 4 (Ubuntu 18.04) | `ultralytics/ultralytics:latest-jetson-jetpack4` |
