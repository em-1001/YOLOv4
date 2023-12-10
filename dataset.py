import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height,
    nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13,26,52],
        C=20,
        transform=None
        ):

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # [class,x,y,w,h] -> [x,y,w,h,class]
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]


        # Ground Truth ..?
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 -> [objectness, x, y, w, h, class]

        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale  # scale -> 0, 1, 2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # an anchor of particular scale -> 0, 1, 2
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x) # e.g. x = 0.5, S = 13 --> int(6.5) = 6th cell of x
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # objectness = 1
                    x_cell, y_cell = S*x - j, S*y - i # both are between [0, 1] e.g. 6.5 - 6 = 0.5

                    width_cell, height_cell = (       # ground truth에 w, h를 log(bw/pw), log(bh/ph)로 넣지 않는 이유..?
                        width * S,  # e.g. S = 13, width = 0.5, 6.5
                        height * S
                    )

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:  # box가 현재 scale_idx에 대한 anchor 할당을 iou가 가장 높은거로 받았으나 각 scale당 3개의 anchor가 존재하므로 iou가 가장 높은거 이외의 것들이 여기로 온다?
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this predicion

        return image, tuple(targets)
    
    
def test():
    anchors = config.ANCHORS

    transform = config.test_transform

    dataset = YOLODataset(
        "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/train.csv",
        "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/images/",
        "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    cnt = 0


    for x, y in loader:
        cnt += 1
        if cnt == 4:
            break

        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        print(len(boxes))
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()