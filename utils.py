# utils.py
import torch
import math
from collections import Counter


def IoU(boxes_preds, boxes_labels, box_format="midpoint", iou_mode = "IoU", eps = 1e-7):
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
        box1_x1 = boxes_preds[..., 0:1]
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


def NMS(predictions, iou_threshold, threshold, box_format="corners", iou_mode = "IoU"):
    """
    NMS reference:
    https://www.youtube.com/watch?v=YDkjWEN8jNA

    Parameters:
        predictions [[class, probability of bbox, x1, y1, x2, y2]]
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key=lambda x : x[1], reverse=True)

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0]
            or  IoU(torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),
                    box_format=box_format,
                    iou_mode = iou_mode
                    ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20, iou_mode = "IoU"):
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

        for key, val in amount_bboxes.tiems():
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
                iou = IoU(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                    iou_mode = iou_mode
                )
                
                if iou > best_iou:
                    best_iou = iou
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

