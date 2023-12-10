import config
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import YOLOv3
from dataset import YOLODataset
from tqdm import tqdm
from utils import (
    mAP,
    nms,
    cells_to_bboxes,
    get_evaluation_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YOLOLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                  loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)

# Creating the model from YOLOv3 class
model = YOLOv3().to(config.DEVICE)

# Defining the optimizer
optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

# Defining the loss function
loss_fn = YOLOLoss()

# Defining the scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Defining the train dataset
train_dataset = YOLODataset(
    csv_file="/content/PASCAL_VOC/train.csv",
    img_dir="/content/PASCAL_VOC/images/",
    label_dir="/content/PASCAL_VOC/labels/",
    anchors=config.ANCHORS,
    transform=config.train_transform
)

# Defining the train data loader
train_loader = DataLoader(
    train_dataset,
    batch_size = config.BATCH_SIZE,
    num_workers = 4,
    shuffle = True,
    pin_memory = True,
)

# Scaling the anchors
scaled_anchors = (
    torch.tensor(config.ANCHORS) * 
    torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(config.DEVICE)


LOAD_MODEL = False
# Loading the checkpoint 
if LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

# Training the model
for e in range(1, config.EPOCHS+1):
    print("Epoch:", e)
    training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

    # Saving the model
    if config.SAVE_MODEL:
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

# Taking a sample image and testing the model

# Setting the load_model to True
LOAD_MODEL = True

# Defining the model, optimizer, loss function and scaler
model = YOLOv3().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
loss_fn = YOLOLoss()
scaler = torch.cuda.amp.GradScaler()

# Loading the checkpoint
if LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

# Defining the test dataset and data loader
test_dataset = YOLODataset(
    csv_file="/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/test.csv",
    img_dir="/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/images/",
    label_dir="/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/labels/",
    anchors=config.ANCHORS,
    transform=config.test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size = config.BATCH_SIZE,
    num_workers = 4,
    shuffle = True,
)

# Getting a sample image from the test data loader
x, y = next(iter(test_loader))
x = x.to(config.DEVICE)

model.eval()
with torch.no_grad():
    # Getting the model predictions
    output = model(x)
    # Getting the bounding boxes from the predictions
    bboxes = [[] for _ in range(x.shape[0])]
    anchors = (
            torch.tensor(config.ANCHORS)
                * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
            ).to(config.DEVICE)

    # Getting bounding boxes for each scale
    for i in range(3):
        batch_size, A, S, _, _ = output[i].shape
        anchor = anchors[i]
        boxes_scale_i = cells_to_bboxes(
                            output[i], anchor, S=S, is_preds=True
                        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
model.train()

print(len(bboxes))
# Plotting the image with bounding boxes for each image in the batch
for i in range(config.BATCH_SIZE):
    # Applying non-max suppression to remove overlapping bounding boxes
    print("bboxes : " + str(len(bboxes[i])))
    nms_boxes = nms(bboxes[i], iou_threshold=0.4, threshold=0.6, iou_mode = "DIoU")
    print("nms_bboxes : " + str(len(nms_boxes)))
    # Plotting the image with bounding boxes
    plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
  

# mAP test 
pred_boxes, true_boxes = get_evaluation_bboxes(
    test_loader,
    model,
    iou_threshold=0.45, # NMS_IOU_THRESH
    anchors=config.ANCHORS,
    threshold=0.2,      # CONF_THRESHOLD
)

# get_evaluation_bboxes 
mapval = mAP(
    pred_boxes,
    true_boxes,
    iou_threshold=config.MAP_IOU_THRESH,
    box_format="midpoint",
    num_classes=config.NUM_CLASSES,
)

print(f"MAP: {mapval.item()}")

