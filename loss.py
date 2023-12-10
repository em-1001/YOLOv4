import random
import torch
import torch.nn as nn

from utils import iou

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()


        # constants 
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # predictions -> [objectness, x, y, w, h, class] ..?

        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        
        # No object Loss
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])
        )

        
        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) # 3(anchor) x 2(h, w), p_w * exp(t_w)를 연산하기 위해 reshape
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj])) # target과 iou로 겹치는 정도만 gt로 반영

        
        # Box Coordinate Loss 
        # object detection box loss iou 검색 -> https://learnopencv.com/iou-loss-functions-object-detection/
        
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y to be between [0, 1]
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        
        
        """
        anchors = anchors.reshape(1, 3, 1, 1, 2) # 3(anchor) x 2(h, w), p_w * exp(t_w)를 연산하기 위해 reshape
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        iou_loss = 1 - iou(boxes_preds=box_preds[obj], boxes_labels=target[..., 1:5][obj], box_format="midpoint", iou_mode = "CIoU")
        box_loss = sum(iou_loss)/len(iou_loss)
        # box_loss = iou_loss.mean()"""


        # Class Loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )