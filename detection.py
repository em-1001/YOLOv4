import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import config
import torch
import torch.optim as optim
from PIL import Image, ImageFile

from yolov4 import YOLOv4
from tqdm import tqdm
from utils import (
    mean_average_precision,
    non_max_suppression,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    plot_image
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ]
)

model = YOLOv4(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)
loss_fn = YoloLoss()
scaler = torch.cuda.amp.GradScaler()

if True: # config.LOAD_MODEL
    load_checkpoint(
        "/content/CIoU100epoch.pth.tar", model, optimizer, config.LEARNING_RATE
    )

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)


model.eval()
img_path = "/content/f22.jpg"
image = np.array(Image.open(img_path).convert("RGB"))
augmentations = test_transforms(image=image)
image = augmentations["image"]
x = image.to("cuda")
print(x.shape)
with torch.no_grad():
    out = model(x.unsqueeze(0))
    bboxes = [[] for _ in range(x.shape[0])]
    for i in range(3):
        batch_size, A, S, _, _ = out[i].shape
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(
            out[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

nms_boxes = non_max_suppression(
    bboxes[0], iou_threshold=0.45, threshold=0.65, box_format="midpoint", iou_mode="DIoU"
)
plot_image(x.permute(1,2,0).detach().cpu(), nms_boxes)
