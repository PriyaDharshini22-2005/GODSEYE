import os
import torch
import torchvision.models as models
from ultralytics import YOLO
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import boxes as box_ops

# Function to read .config file
def parse_config(file_path):
    config = {}
    with open(file_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                config[key.strip()] = value.strip()
    return config

# Load config values
config_path = ".config"
config = parse_config(config_path)

# Get paths from config
ROOT_FOLDER = r"D:\PROJECTS\GODSEYE\Raw_data\Raw_data" # Root folder path
WEIGHTS11 = os.path.normpath(config.get("WEIGHTS11", "").strip())      # YOLO weights path

# Print debug information
print(f"ROOT_FOLDER from config: {ROOT_FOLDER}")
print(f"WEIGHTS11 from config: {WEIGHTS11}")

# Ensure WEIGHTS11 is an absolute path
if not os.path.isabs(WEIGHTS11):
    WEIGHTS11 = os.path.join(os.getcwd(), WEIGHTS11)
WEIGHTS11 = os.path.normpath(WEIGHTS11)

# Ensure ROOT_FOLDER is an absolute path
if not os.path.isabs(ROOT_FOLDER):
    ROOT_FOLDER = os.path.join(os.getcwd(), ROOT_FOLDER)
ROOT_FOLDER = os.path.normpath(ROOT_FOLDER)

# Validate paths
if not os.path.exists(ROOT_FOLDER):
    raise FileNotFoundError(f"Root folder not found: {ROOT_FOLDER}")
if not os.path.exists(WEIGHTS11):
    raise FileNotFoundError(f"YOLO weights file not found: {WEIGHTS11}")

# Define log directory
log_dir = r"D:\PROJECTS\GODSEYE\logsfolder\devanadhan"
os.makedirs(log_dir, exist_ok=True)

# Define output directory for processed images
output_img_dir = os.path.join(log_dir, "output_imgs")
os.makedirs(output_img_dir, exist_ok=True)

class ResNetBackbone(torch.nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.activation = {}

        self.resnet.conv1.register_forward_hook(self.get_activation('conv1', 'backbone'))
        self.resnet.layer1.register_forward_hook(self.get_activation('layer1', 'backbone'))
        self.resnet.layer2.register_forward_hook(self.get_activation('layer2', 'backbone'))
        self.resnet.layer3.register_forward_hook(self.get_activation('layer3', 'backbone'))
        self.resnet.layer4.register_forward_hook(self.get_activation('layer4', 'backbone'))

    def get_activation(self, name, module_name):
        def hook(module, input, output):
            self.activation[f"{module_name}_{name}"] = output
            feature_map = output.detach().cpu().numpy()

            # Save statistics to log file
            log_file = os.path.join(log_dir, "feature_map_stats.txt")
            with open(log_file, "a") as f:
                f.write(f"\nFeature Information for {module_name} {name}:\n")
                f.write(f"Shape: {output.shape}\n")
                f.write(f"Min: {np.min(feature_map):.3f}, Max: {np.max(feature_map):.3f}, "
                        f"Mean: {np.mean(feature_map):.3f}, Std: {np.std(feature_map):.3f}\n")

        return hook

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class BiFPN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = {}

        # Register hooks
        self.conv1.register_forward_hook(self.get_activation('conv1', 'neck'))
        self.conv2.register_forward_hook(self.get_activation('conv2', 'neck'))
        self.conv3.register_forward_hook(self.get_activation('conv3', 'neck'))

    def get_activation(self, name, module_name):
        def hook(module, input, output):
            self.activation[f"{module_name}_{name}"] = output
            feature_map = output.detach().cpu().numpy()

            # Save statistics to log file
            log_file = os.path.join(log_dir, "feature_map_stats.txt")
            with open(log_file, "a") as f:
                f.write(f"\nFeature Information for {module_name} {name}:\n")
                f.write(f"Shape: {output.shape}\n")
                f.write(f"Min: {np.min(feature_map):.3f}, Max: {np.max(feature_map):.3f}, "
                        f"Mean: {np.mean(feature_map):.3f}, Std: {np.std(feature_map):.3f}\n")

        return hook

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return torch.cat((x1, x2, x3), dim=1)

class YOLOHead(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOHead, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.swish1 = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(in_channels, num_classes + 5, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = {}

        # Register hooks
        self.conv1.register_forward_hook(self.get_activation('conv1', 'head'))
        self.conv2.register_forward_hook(self.get_activation('conv2', 'head'))

    def get_activation(self, name, module_name):
        def hook(module, input, output):
            self.activation[f"{module_name}_{name}"] = output
            feature_map = output.detach().cpu().numpy()

            # Save statistics to log file
            log_file = os.path.join(log_dir, "feature_map_stats.txt")
            with open(log_file, "a") as f:
                f.write(f"\nFeature Information for {module_name} {name}:\n")
                f.write(f"Shape: {output.shape}\n")
                f.write(f"Min: {np.min(feature_map):.3f}, Max: {np.max(feature_map):.3f}, "
                        f"Mean: {np.mean(feature_map):.3f}, Std: {np.std(feature_map):.3f}\n")

        return hook

    def forward(self, x):
        x = self.swish1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

def visualize_and_save_feature_maps(feature_maps, layer_name):
    num_feature_maps = feature_maps.shape[1]
    fig, axes = plt.subplots(1, min(8, num_feature_maps), figsize=(20, 20))
    fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16)

    if num_feature_maps == 1:
        axes = [axes]

    for i in range(min(8, num_feature_maps)):
        axes[i].imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        axes[i].axis('off')

    output_path = os.path.join(log_dir, f"{layer_name}.png")
    plt.savefig(output_path)
    plt.close()

def detect_objects(image_path, model):
    img = Image.open(image_path).convert('RGB')
    transform = Compose([Resize((640, 640)), ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        results = model(img_tensor)

    orig_img = cv2.imread(image_path)
    orig_h, orig_w = orig_img.shape[:2]

    pred = results[0].boxes
    boxes = pred.xyxy.cpu()
    scores = pred.conf.cpu()
    classes = pred.cls.cpu()

    conf_threshold = 0.2
    keep = scores > conf_threshold
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    keep_indices = box_ops.batched_nms(boxes, scores, torch.zeros(len(boxes)), 0.4)
    boxes, scores, classes = boxes[keep_indices], scores[keep_indices], classes[keep_indices]

    boxes[:, [0, 2]] *= orig_w / 640
    boxes[:, [1, 3]] *= orig_h / 640

    img_display = orig_img.copy()
    detection_summary = {}

    # Define colors for each class
    class_colors = {
        "dent": (255, 0, 0),    # Red
        "peeloff": (0, 255, 0), # Green
        "scratch": (0, 0, 255)  # Blue
    }

    for box, score, cls_idx in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_name = model.names[int(cls_idx.item())]
        color = class_colors.get(cls_name, (255, 255, 255))  # Default to white if class not found

        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
        label = f'{cls_name}: {score:.2f}'
        cv2.putText(img_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1

    if not detection_summary:
        cv2.putText(img_display, "GOOD PART", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the output image
    output_image_path = os.path.join(output_img_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img_display)

    log_file = os.path.join(log_dir, "detection_summary.txt")
    with open(log_file, "w") as f:
        if detection_summary:
            for cls_name, count in detection_summary.items():
                f.write(f"{cls_name}: {count} detections\n")
        else:
            f.write("GOOD PART\n")

    return detection_summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model = YOLO(WEIGHTS11)
        model.model.backbone = ResNetBackbone().to(device)
        model.model.neck = BiFPN(in_channels=2048, out_channels=256).to(device)
        model.model.head = YOLOHead(in_channels=768, num_classes=80).to(device)
        model.model.act = torch.nn.SiLU().to(device)

        # Define the subfolders to process
        subfolders = ["Cut Mark", "Dent Mark", "Peel Off"]

        for subfolder in subfolders:
            folder_path = os.path.join(ROOT_FOLDER, subfolder)
            if not os.path.exists(folder_path):
                print(f"Subfolder {folder_path} does not exist. Skipping.")
                continue

            # Process each image in the subfolder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if not os.path.isfile(image_path):
                    continue

                print(f"Processing image: {image_path}")

                # Process the image
                detect_objects(image_path, model)

        print("\nProcessing complete! Results saved in the output_imgs folder.")

    except Exception as e:
        with open(os.path.join(log_dir, "error_log.txt"), "w") as f:
            f.write(str(e))
        print("Error occurred, check logs for details.")

if __name__ == "__main__":
    main()