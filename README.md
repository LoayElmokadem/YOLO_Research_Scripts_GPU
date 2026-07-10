# YOLO Benchmark Suite — NVIDIA Jetson Nano & Jetson Orin Nano

A comprehensive benchmarking toolkit for evaluating **YOLO object detection models (v8–v12)** on resource-constrained edge hardware. This suite measures real-world performance across five key dimensions: **model load time, inference latency & FPS, CPU/RAM usage, disk I/O, and detection accuracy (mAP & IoU)** — with full thermal monitoring via `tegrastats`.

> **Target Hardware:** NVIDIA Jetson Nano (JetPack 4) · Jetson Orin Nano (JetPack 5/6) | **Dataset:** COCO128

---

## 📊 Benchmark Metrics

| Metric | Script | Description |
|---|---|---|
| Model Load Time | `load_time.py` | Time (seconds) to load the `.pt` model into memory |
| Disk I/O | `disk5.py` | Avg read/write bytes and speed (MB/s) during inference |
| CPU, RAM & Temperature | `ram_cpu.py` | CPU %, memory %, FPS, latency, GPU and CPU core temperatures |
| mAP / Precision / Recall | `validation.py` | YOLO `.val()` on COCO128 |
| IoU | `iou.py` | Per-image IoU vs. COCO128 ground truth labels |

---

## 📁 Repository Structure

```
YOLO_RESEARCH_GPU_REPO/
│
├── # ── Benchmark Scripts ──────────────────────────────────────
├── load_time.py          # Measures model load time
├── disk5.py              # Measures disk I/O throughput during inference
├── ram_cpu.py            # CPU, RAM, FPS, latency, and thermal monitoring
├── validation.py         # Runs YOLO .val() → mAP, Precision, Recall
├── iou.py                # Per-image IoU vs. COCO128 ground truth
├── gpu.py                # Quick GPU utilization snapshot (gpustat)
│
├── # ── Orchestration ──────────────────────────────────────────
├── all_scripts2.sh       # Main runner: benchmarks all model sizes in sequence
├── clear_cache.sh        # Clears FS, Python, and GPU caches between runs
│
├── # ── Config ─────────────────────────────────────────────────
├── coco17.yaml           # COCO 2017 class definitions for validation
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes weights, datasets, logs, and outputs
│
├── # ── Docker ─────────────────────────────────────────────────
├── docker_setup.md       # Docker setup guide for Jetson Nano & Jetson Orin Nano
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/YOLO_RESEARCH_GPU_REPO.git
cd YOLO_RESEARCH_GPU_REPO
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Jetson users:** Follow [docker_setup.md](docker_setup.md) for device-specific Docker setup — covers both the Jetson Nano (JetPack 4) and Jetson Orin Nano (JetPack 5/6) using official Ultralytics images.

### 3. Prepare the COCO128 Dataset

The benchmark scripts expect the COCO128 dataset at `coco128/images/train2017/` and `coco128/labels/train2017/`.

```bash
# Download via Ultralytics (auto-downloads on first model run)
python3 -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco128.yaml')"
```

Or manually download from [ultralytics/assets](https://github.com/ultralytics/assets/releases).

---

## 🚀 Usage

### Run the Full Benchmark Suite (All Model Sizes)

Edit the `models` array in `all_scripts2.sh` to select the YOLO family and sizes you want:

```bash
# Examples — uncomment the line for the series you want to benchmark:
# models=("8n" "8s" "8m" "8l" "8x")     # YOLOv8
# models=("9t" "9s" "9m" "9c" "9e")     # YOLOv9
# models=("10n" "10s" "10m" "10l" "10x") # YOLOv10
# models=("11n" "11s" "11m" "11l" "11x") # YOLO11
models=("12n" "12s" "12m" "12l" "12x")   # YOLO12 (default)
```

Then run:

```bash
chmod +x all_scripts2.sh
bash all_scripts2.sh
```

Logs are saved per model in `logs/<model_size>.log`.

---

### Run Individual Scripts

Each script reads the target model size from the `YOLO_MODEL_SIZE` environment variable (defaults to `"11n"` if not set):

```bash
# Example: benchmark YOLO12n
YOLO_MODEL_SIZE="12n" python3 load_time.py
YOLO_MODEL_SIZE="12n" python3 disk5.py
YOLO_MODEL_SIZE="12n" python3 ram_cpu.py
YOLO_MODEL_SIZE="12n" python3 validation.py
YOLO_MODEL_SIZE="12n" python3 iou.py
```

---

### Script Details

#### `load_time.py`
Measures wall-clock time to load the YOLO model weights into memory.
```
Model load time: 1.243 seconds
```

#### `disk5.py`
Processes each image in COCO128 and records per-image disk read/write bytes and speed.
```
Summary for all images:
Total Images: 128
Average Elapsed Time: 0.0521 seconds
Average Disk Read Speed: 3.8412 MB/s
Average Disk Write Speed: 1.2047 MB/s
```

#### `ram_cpu.py`
The primary benchmark. Measures inference across all COCO128 images and reports:
```
Average CPU Usage (script-specific): 12.34%
Average Memory Usage (script-specific): 3.21%
Average latency: 48.72 ms
FPS: 20.53
Temp (min) GT: 38.50C
Temp (max) GT: 52.00C
Temp (min) Cores: 36.00C
Temp (max) Cores: 49.50C
```

#### `validation.py`
Runs the official YOLO validation pipeline. Outputs COCO-style metrics:
```
all    128   929   0.612   0.491   0.534   0.351
```
*(columns: images, instances, Precision, Recall, mAP50, mAP50-95)*

#### `iou.py`
Computes the average IoU between YOLO predictions and COCO128 ground truth bounding boxes:
```
Average IoU across all images: 0.6812
```

#### `gpu.py`
Quick snapshot of GPU utilization using `gpustat`:
```bash
python3 gpu.py
# GPU 0: NVIDIA Tegra X1, Utilization: 87%
```

---

## 📋 Supported Models

| Family | Sizes |
|--------|-------|
| YOLOv8 | n, s, m, l, x |
| YOLOv9 | t, s, m, c, e |
| YOLOv10 | n, s, m, l, x |
| YOLO11  | n, s, m, l, x |
| YOLO12  | n, s, m, l, x |

Model weights are downloaded automatically by Ultralytics on first run.

---

## 🌡️ Temperature Monitoring

On Jetson Nano, thermal data is read via `tegrastats`. The `all_scripts2.sh` orchestrator also records **start and end GPU temperatures** for each model run, and clears all caches between runs using `clear_cache.sh` to ensure measurement isolation.

---

## 🐳 Docker (Jetson Nano & Jetson Orin Nano)

See [docker_setup.md](docker_setup.md) for the full setup guide covering both the **Jetson Nano** (`ultralytics/ultralytics:latest-jetson-jetpack4`) and **Jetson Orin Nano** (`ultralytics/ultralytics:latest-jetson-jetpack5` or `jetpack6`).

---

## 📦 Output

Results are saved as plain-text logs in the `logs/` directory (one file per model size). Example: `logs/12n.log`.

> **Note:** The `logs/` and `results/` directories are excluded from version control via `.gitignore`. Commit only the scripts and configuration.

---

## 📄 License

This project is intended for academic research. Please cite appropriately if used in publications.
