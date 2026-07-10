import os
import cv2
from ultralytics import YOLO

def yolo_to_pixel(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return [x_min, y_min, x_max, y_max]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union

image_folder = r"coco128/images/train2017"
label_folder = r"coco128/labels/train2017"

model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolov{model_size}.pt"

try:
    model = YOLO(model_file)  # Repla
except:
    model_file = f"yolo{model_size}.pt"
    model = YOLO(model_file)

iou_results = []
total_iou = 0
num_predictions = 0

for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".png")):
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_folder, label_file)
        if not os.path.exists(label_path):
            print(f"Label file missing for {image_file}. Skipping...")
            continue

        ground_truth_boxes = []
        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                class_id = int(data[0])  # Class ID
                bbox = list(map(float, data[1:]))
                ground_truth_boxes.append(yolo_to_pixel(bbox, img_width, img_height))

        print(f"Image: {image_file} - Ground Truth Boxes: {ground_truth_boxes}")

        results = model.predict(img_path, save=False, verbose=False)
        predictions = results[0].boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]

        print(f"Image: {image_file} - Predicted Boxes: {predictions}")

        for pred_box in predictions:
            best_iou = 0
            for gt_box in ground_truth_boxes:
                iou = calculate_iou(pred_box, gt_box)
                best_iou = max(best_iou, iou)

            iou_results.append((image_file, pred_box.tolist(), best_iou))
            total_iou += best_iou
            num_predictions += 1

average_iou = total_iou / num_predictions if num_predictions > 0 else 0
print(f"Average IoU across all images: {average_iou:.4f}")

# # Print IoU results
# for image_file, pred_box, iou in iou_results:
#     print(f"Image: {image_file}, Prediction: {pred_box}, IoU: {iou:.4f}")

# # Optionally, save IoU results to a file
# with open("iou_results.txt", "w") as f:
#     f.write(f"Average IoU: {average_iou:.4f}\n")
#     for image_file, pred_box, iou in iou_results:
#         f.write(f"Image: {image_file}, Prediction: {pred_box}, IoU: {iou:.4f}\n")
