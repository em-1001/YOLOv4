import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import math

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def iou(boxes_preds, boxes_labels, box_format="midpoint", iou_mode = "IoU", eps = 1e-7):
    """
    iou Reference:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2  # mid_x - width/2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2  # mid_y - height/2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2  # mid_x + width/2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2  # mid_y + height/2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2


    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]  # 굳이 0:1로 하는 이유는 차원을 유지하기 위해서다.
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # width, height of predict and ground truth box
    w1, h1 = box1_x2 - box1_x1, box1_y2 - box1_y1 + eps
    w2, h2 = box2_x2 - box2_x1, box2_y2 - box2_y1 + eps

    # coordinates for intersection
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # width, height of convex(smallest enclosing box)
    C_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
    C_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)

    # convex diagonal squared
    c2 = C_w**2 + C_h**2 + eps

    # center distance squared
    p2 = ((box2_x1 + box2_x2 - box1_x1 - box1_x2)**2 + (box2_y1 + box2_y2 - box1_y1 - box1_y2)**2) / 4

    # iou
    box1_area = abs(w1 * h1)
    box2_area = abs(w2 * h2)
    union = box1_area + box2_area - intersection + eps

    iou = intersection / union

    if iou_mode == "GIoU":
        C = abs(C_w * C_h) + eps
        B = abs(w1 * h1)
        B_gt = abs(w2 * h2)

        R_giou = abs(C - (B + B_gt - intersection)) / abs(C)

        return iou - R_giou

    elif iou_mode == "DIoU":
        R_diou = p2 / c2

        return iou - R_diou

    elif iou_mode == "CIoU":
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 - iou) + v + eps)

        R_ciou = p2 / c2 + v * alpha

        return iou - R_ciou

    else:
        return iou
    
    

def nms(bboxes, iou_threshold, threshold, box_format='midpoint', iou_mode = "IoU"):


    assert type(bboxes) == list


    bboxes = [box for box in bboxes if box[1] > threshold]

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nmn = []

    while bboxes:

        chosen_box = bboxes.pop(0)

        
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] 
               or iou(torch.tensor(chosen_box[2:]),
                      torch.tensor(box[2:]),
                      box_format=box_format,
                      iou_mode=iou_mode
                     ) < iou_threshold]
        # bboxes = [box for box in bboxes if box[0] != chosen_box[0]]

        bboxes_after_nmn.append(chosen_box)

    return bboxes_after_nmn




# 이미지 크기의 몇 퍼센트 지점에 x, y 좌표가 있고, 이미지 크기의 몇 퍼센트의 크기로 w, h가 있는지를 반환하는 함수
# grid-relative values to image-relative values
def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates

    [objectness, x, y, w, h, class_num] -> [best_class, objectness, x, y, w, h]
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])

        # predictions[..., 5:]는 각 클래스에 대한 예측 확률 값이고 여기에 argmax를 취하여 모델이 예측한 class 중 가장 확률이 높은 것을 뽑아낸다.
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) # predictions[..., 5:] 어떤 식인지 출력해보기
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        
        if int(class_pred) > 19:
            class_pred = 0
        
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)] + " " + str(round(box[1], 2)),
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()

    

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval() # no gradients are computed during evaluation mode
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i): # box is tensor of [class_pred, prob_score, x1, y1, x2, y2]
                bboxes[idx] += box # 디버깅 필요!! box가 어케 +=으로 들어가는지 확인해야 함

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


    
def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20, iou_mode = "IoU"):
    """
    mAP reference:
    https://www.youtube.com/watch?v=FppOzcDvaDI
    https://ctkim.tistory.com/entry/mAPMean-Average-Precision-%EC%A0%95%EB%A6%AC

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold : Defaultly for mAP @ 50 IoU


    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        detections.sort(key=lambda x : x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iouu = iou(torch.tensor(detection[3:]),
                          torch.tensor(gt[3:]),
                          box_format=box_format,
                          iou_mode = iou_mode)

                if iouu > best_iou:
                    best_iou = iouu
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        recalls = torch.cat((torch.tensor([0]), recalls))       # x axis
        precisions = torch.cat((torch.tensor([1]), precisions)) # y axis
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
    

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr    