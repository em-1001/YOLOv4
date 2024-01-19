# refernce : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/train.py

import config
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, box_loss="MSE"):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0], box_loss=box_loss)
                + loss_fn(out[1], y1, scaled_anchors[1], box_loss=box_loss)
                + loss_fn(out[2], y2, scaled_anchors[2], box_loss=box_loss)
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # if True : #config.SCHEDULER == "CosineAnnealingLR":
        #    scheduler.step()

        scaler.update()



        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def func(epoch):
    if epoch < 30:
        return 1     # 0.0001
    elif epoch < 50:
        return 0.5   # 0.00005
    else:
        return 0.2   # 0.00001


def main():
    model = YOLOv4(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=1e-4
    )
    if True: #config.SCHEDULER == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if False: # config.LOAD_MODEL
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(100):
        print(scheduler._last_lr)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, box_loss="CIoU")

        scheduler.step()
        if True: # config.SAVE_MODEL
            save_checkpoint(model, optimizer, filename=f"/content/drive/MyDrive/yolov3/checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and (epoch+1) % 10 == 0:
            # plot_couple_examples(model, test_loader, 0.4, 0.45, scaled_anchors , iou_mode="IoU")
            # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=0.45,
                anchors=config.ANCHORS,
                threshold=0.5,
                iou_mode = "IoU"
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()
