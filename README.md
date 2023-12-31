

# YOLOv3
## Bounding Box
<p align="center"><img src="https://github.com/em-1001/YOLOv3/blob/master/image/bbox.png">


YOLOv2 부터 Anchor box(prior box)를 미리 설정하여 최종 bounding box 예측에 활용한다. 위 그림에서는 $b_x, b_y, b_w, b_h$가 최종적으로 예측하고자 하는 bounding box이다. 검은 점선은 사전에 설정된 Anchor box로 이 Anchor box를 조정하여 파란색의 bounding box를 예측하도록 한다.   

모델은 직접적으로 $b_x, b_y, b_w, b_h$를 예측하지 않고 $t_x, t_y, t_w, t_h$를 예측하게 된다. 
범위제한이 없는 $t_x, t_y$에 sigmoid($\sigma$)를 적용해주어 0과 1사의 값으로 만들어주고, 이를 통해 bbox의 중심 좌표가 1의 크기를 갖는 현재 cell을 벗어나지 않도록 해준다. 여기에 offset인 $c_x, c_y$를 더해주면 최종적인 bbox의 중심 좌표를 얻게 된다.    
$b_w, b_h$의 경우 미리 정해둔 Anchor box의 너비와 높이를 얼만큼의 비율로 조절할 지를 Anchor와 $t_w, t_h$에 대한 log scale을 이용해 구한다. 

YOLOv2에서는 bbox를 예측할 때 $t_x, t_y, t_w, t_h$를 예측한 후 그림에서의 $b_x, b_y, b_w, b_h$로 변형한 뒤 $L_2$ loss를 통해 학습시켰지만, YOLOv3에서는 ground truth의 좌표를 거꾸로 $\hat{t}_ {∗}$로 변형시켜 예측한 $t_{∗}$와 직접 $L_1$ loss로 학습시킨다. ground truth의 $x, y$좌표의 경우 아래와 같이 변형되고, 

$$
\begin{aligned}
&b_{∗}= \sigma(\hat{t}_ {∗}) + c_{∗}\\      
&\sigma(\hat{t}_ {∗}) = b_{∗} - c_{∗}\\      
&\hat{t}_ {∗} = \sigma^{-1}(b_{∗} - c_{∗})
\end{aligned}$$

$w, h$는 아래와 같이 변형된다. 

$$\hat{t}_ {∗} = \ln\left(\frac{b_{∗}}{p_{∗}}\right)$$

결과적으로 $x, y, w, h$ loss는 ground truth인 $\hat{t}_ {∗}$ prediction value인 ${t}_ {∗}$사이의 차이 $\hat{t}_ {∗} - {t}_ {∗}$를 통한 Sum-Squared Error(SSE)로 구해진다. 

## Model

<p align="center"><img src="https://github.com/em-1001/YOLOv3/blob/master/image/darknet53.png" height="35%" width="35%"><img src="https://github.com/em-1001/YOLOv3/blob/master/image/model1.png" height="65%" width="65%"></p>


모델의 backbone은 $3 \times 3$, $1 \times 1$ Residual connection을 사용하면서 최종적으로 53개의 conv layer를 사용하는 **Darknet-53** 을 이용한다. Darknet-53의 Residual block안에서도 bottleneck 구조를 사용하며, input의 channel을 중간에 반으로 줄였다가 다시 복구시킨다. 이때 Residual block의 $1 \times 1$ conv는 $s=1, p=0$ 이고, $3 \times 3$ conv는 $s=1, p=1$이다. 

YOLOv3 model의 특징은 물체의 scale을 고려하여 3가지 크기의 output이 나오도록 FPN과 유사하게 설계하였다는 것이다. 오른쪽 그림과 같이 $416 \times 416$의 크기를 feature extractor로 받았다고 하면, feature map이 크기가 $52 \times 52$, $26 \times 26$, $13 \times 13$이 되는 layer에서 각각 feature map을 추출한다. 

<img src="https://github.com/em-1001/YOLOv3/blob/master/image/model2.png"> 

그 다음 가장 높은 level, 즉 해상도가 가장 낮은 feature map부터 $1 \times 1$, $3 \times 3$ conv layer로 구성된 작은 Fully Convolutional Network(FCN)에 입력한다. 이후 이 FCN의 output channel이 512가 되는 시점에서 feature map을 추출한 뒤, $2\times$로 upsampling을 진행한다. 이후 바로 아래 level에 있는 feature map과 concatenate를 해주고, 이렇게 만들어진 merged feature map을 다시 FCN에 입력한다. 이 과정을 다음 level에도 똑같이 적용해주고 이렇게 3개의 scale을 가진 feature map이 만들어진다. 각 scale에 따라 나오는 최종 feature map의 형태는 $N \times N \times \left[3 \cdot (4+1+80)\right]$이다. 여기서 $3$은 grid cell당 predict하는 anchor box의 수를, $4$는 bounding box offset $(x, y, w, h)$, $1$은 objectness prediction, $80$은 class의 수 이다. 따라서 최종적으로 얻는 feature map은 $\left[52 \times 52 \times 255\right], \left[26 \times 26 \times 255\right], \left[13 \times 13 \times 255\right]$이다. 

이러한 방법을 통해 더 높은 level의 feature map으로부터 fine-grained 정보를 얻을 수 있으며, 더 낮은 level의 feature map으로부터 더 유용한 semantic 정보를 얻을 수 있다.



## Loss Function

$$λ_ {coord} \sum_ {i=0}^{S^2} \sum_ {j=0}^B 𝟙^{obj}_ {i j} \left[(t_ {x_ i} - \hat{t_ {x_ i}})^2 + (t_ {y_ i} - \hat{t_ {y_ i}})^2 \right]$$

$$+λ_ {coord} \sum_ {i=0}^{S^2} \sum_ {j=0}^B 𝟙^{obj}_ {i j} \left[(t_ {w_ i} - \hat{t_ {w_ i}})^2 + (t_ {h_ i} - \hat{t_ {h_ i}})^2 \right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[-(o_i\log(\hat{o_i}) + (1 - o_i)\log(1 - \hat{o_i}))\right]$$

$$+Mask_{ig} \cdot λ_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{noobj}_{i j} \left[-(o_i\log(\hat{o_i}) + (1 - o_i)\log(1 - \hat{o_i}))\right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_ {i j} \sum_{c \in classes} \left[-(c_i\log(\hat{c_i}) + (1 - c_i)\log(1 - \hat{c_i}))\right]$$  

$S$ : number of cells    
$B$ : number of anchors  
$o$ : objectness  
$c$ : class label  
$λ_ {coord}$ : coordinate loss balance constant  
$λ_{noobj}$ : no confidence loss balance constant    
$𝟙^{obj}_ {i j}$ : 1 when there is object, 0 when there is no object  
$𝟙^{noobj}_ {i j}$ : 1 when there is no object, 0 when there is object  
$Mask_{ig}$ : tensor that masks only the anchor with iou $\le$ 0.5. Have a shape of $\left[S, S, B\right]$.

각각의 box는 multi-label classification을 하게 되는데 논문에서는 softmax가 성능이 좋지 못하기 때문에, binary cross-entropy loss를 사용했다고 한다. 하나의 box안에 복수의 객체가 존재하는 경우 softmax는 적절하게 객체를 알아내지 못하기 때문에, box 안에 각 class가 존재하는 여부를 확인하는 binary cross-entropy가 보다 적절하다고 할 수 있다.   

$o$ (objectness)는 anchor와 bbox의 iou가 가장 큰 anchor의 값이 1, 그렇지 않은 경우의 값이 0인 $\left[N, N, 3, 1\right]$의 tensor로 만들어진다. $c$ (class label)은 one-encoding으로 $\left[N, N, 3, n \right]$ ($n$ : num_classes) 의 shape를 갖는 tensor로 만들어진다. 


# YOLOv4
## Model
<p align="center"><img src="https://github.com/em-1001/CSPDarknet53-SPP/blob/master/image/CSPDarknet53.png" height="55%" width="55%"></p>

전체적인 구조는 YOLOv3과 유사하지만 YOLOv4는 **CSPDarknet53+SPP**를 사용한다. CSPDarknet53은 Darknet53에 CSPNet을 적용한 것이다. CSPNet은 위 사진의 CSP Residual 부분과 같이 base layer의 feature map을 두 개로 나눈 뒤($X_0 \to X_0^{'}, X_0^{''}$) $X_0^{''}$는 Dense Layer에 통과 시키고 $X_0^{'}$는 그대로 가져와서 마지막에 Dense Layer의 출력값인 ($X_0^{''}, x_1, x_2, ...$)을 transition layer에 통과시킨 $X_T$와 concat시킨다. 이후 concat된 결과가 다음 transition layer를 통과하면서 $X_U$가 생성된다.

$$
\begin{aligned}
X_k &= W_K^{ * }[X_0^{''}, X_1, ..., X_{k-1}]\\  
X_T &= W_T^{ * }[X_0^{''}, X_1, ..., X_{k}]\\    
X_U &= W_U^{ * }[X_0^{'}, X_T]\\      
\end{aligned}$$  
</br>

$$
\begin{aligned}
W_k^{'} &= f(W_k, g_0^{''}, g_1, g_2, ..., g_{k-1})\\  
W_T^{'} &= f(W_T, g_0^{''}, g_1, g_2, ..., g_{k})\\  
W_U^{'} &= f(W_U, g_0^{'}, g_T)\\      
\end{aligned}$$

이렇게 하므로써 CSPDenseNet은 DenseNet의 feature reuse 특성을 활용하면서, gradient flow를 truncate($X_0 \to X_0^{'}, X_0^{''}$)하여 과도한 양의 gradient information 복사를 방지할 수 있다. 


## Box Loss
일반적으로 IoU-based loss는 다음과 같이 표현된다. 

$$L = 1 - IoU + \mathcal{R}(B, B^{gt})$$

여기서 $R(B, B^{gt})$는  predicted box $B$와 target box $B^{gt}$에 대한 penalty term이다.  
$1 - IoU$로만 Loss를 구할 경우 box가 겹치지 않는 case에 대해서 어느 정도의 오차로 교집합이 생기지 않은 것인지 알 수 없어서 gradient vanishing 문제가 발생했다. 이러한 문제를 해결하기 위해 penalty term을 추가한 것이다. 

### Generalized-IoU(GIoU)
Generalized-IoU(GIoU) 의 경우 Loss는 다음과 같이 계산된다. 

$$L_{GIoU} = 1 - IoU + \frac{|C - B ∪ B^{gt}|}{|C|}$$

여기서 $C$는 $B$와 $B^{gt}$를 모두 포함하는 최소 크기의 Box를 의미한다. Generalized-IoU는 겹치지 않는 박스에 대한 gradient vanishing 문제는 개선했지만 horizontal과 vertical에 대해서 에러가 크다. 이는 target box와 수평, 수직선을 이루는 Anchor box에 대해서는 $|C - B ∪ B^{gt}|$가 매우 작거나 0에 가까워서 IoU와 비슷하게 동작하기 때문이다. 또한 겹치지 않는 box에 대해서 일단 predicted box의 크기를 매우 키우고 IoU를 늘리는 동작 특성 때문에 수렴 속도가 느리다. 

### Distance-IoU(DIoU)
GIoU가 면적 기반의 penalty term을 부여했다면, DIoU는 거리 기반의 penalty term을 부여한다. 
DIoU의 penalty term은 다음과 같다. 

$$\mathcal{R}_{DIoU} = \frac{\rho^2(b, b^{gt})}{c^2}$$

$\rho^2$는 Euclidean거리이며 $c$는 $B$와 $B^{gt}$를 포함하는 가장 작은 Box의 대각선 거리이다. 

<p align="center"><img src="https://github.com/em-1001/YOLOv3/blob/master/image/diou.png" height="25%" width="25%"></p>

DIoU Loss는 두 개의 box가 완벽히 일치하면 0, 매우 멀어지면 $L_{GIoU} = L_{DIoU} \to 2$가 된다. 이는 IoU가 0이 되고, penalty term이 1에 가깝게 되기 때문이다. Distance-IoU는 두 box의 중심 거리를 직접적으로 줄이기 때문에 GIoU에 비해 수렴이 빠르고, 거리기반이므로 수평, 수직방향에서 또한 수렴이 빠르다. 

#### DIoU-NMS
DIoU를 NMS(Non-Maximum Suppression)에도 적용할 수 있다. 일반적인 NMS의 경우 이미지에서 같은 class인 두 물체가 겹쳐있는 Occlusion(가림)이 발생한 경우 올바른 박스가 삭제되는 문제가 발생하는데, DIoU를 접목할 경우 두 박스의 중심점 거리도 고려하기 때문에 target box끼리 겹쳐진 경우에도 robust하게 동작할 수 있다. 

$$
s_i =
\begin{cases}
s_ i, & IoU - \mathcal{R}_ {DIoU}(\mathcal{M}, B_i) < \epsilon\\
0, & IoU - \mathcal{R}_{DIoU}(\mathcal{M}, B_i) \ge \epsilon
\end{cases}
$$

가장 높은 Confidence score를 갖는 $\mathcal{M}$에 대해 IoU와 DIoU의 distance penalty를 동시에 고려하여 IoU가 매우 크더라도 중심점 사이의 거리가 멀면 다른 객체를 탐지한 것일 수도 있으므로 위와 같이 일정 임계치 $\epsilon$ 보다 작으면 없애지 않고 보존한다. 


### Complete-IoU(CIoU)
DIoU, CIoU를 제안한 논문에서 말하는 성공적인 Bounding Box Regression을 위한 3가지 조건은 overlap area, central point
distance, aspect ratio이다. 이 중 overlap area, central point는 DIoU에서 이미 고려했고 여기에 aspect ratio를 고려한 penalty term을 추가한 것이 CIoU이다. CIoU penalty term는 다음과 같이 정의된다. 

$$\mathcal{R}_{CIoU} = \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

$$v = \frac{4}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}})^2$$

$$\alpha = \frac{v}{(1 - IoU) + v}$$

$v$의 경우 bbox는 직사각형이고 $\arctan{\frac{w}{h}} = \theta$이므로 $\theta$의 차이를 통해 aspect ratio를 구하게 된다. 이때 $v$에 $\frac{2}{π}$가 곱해지는 이유는 $\arctan$ 함수의 최대치가 $\frac{2}{π}$ 이므로 scale을 조정해주기 위해서이다. 

$\alpha$는 trade-off 파라미터로 IoU가 큰 box에 대해 더 큰 penalty를 주게 된다. 

CIoU에 대해 최적화를 수행하면 아래와 같은 기울기를 얻게 된다. 이때, $w, h$는 모두 0과 1사이로 값이 작아 gradient explosion을 유발할 수 있다. 따라서 실제 구현 시에는 $\frac{1}{w^2 + h^2} = 1$로 설정한다. 

$$\frac{\partial v}{\partial w} = \frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{h}{w^2 + h^2}$$ 

$$\frac{\partial v}{\partial h} = -\frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{w}{w^2 + h^2}$$ 

## Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos{\left(\frac{T_{cur}}{T_{\max}}\pi\right)} \right), \ T_{cur} \neq (2k+1)T_{\max}$$

$$\eta_{t+1} = \eta_{t} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 - \cos{\left(\frac{1}{T_{\max}}\pi\right)} \right), \ T_{cur} = (2k+1)T_{\max}$$

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html   
code : https://github.com/pytorch/pytorch/blob/v1.1.0/torch/optim/lr_scheduler.py#L222

```py
import torch
import torch.optim as optim

class CosineAnnealingLRWithWarmup:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.epoch = 0

    def get_lr(self):
        if self.epoch < self.warmup_epochs:
            return self.epoch / self.warmup_epochs
        else:
            return self.eta_min + 0.5 * (1 + torch.cos((self.epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs) * math.pi))

    def step(self):
        self.epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```
## Mosaic Augmentation

# Performance
<p align="center"><img src="https://github.com/em-1001/YOLOv3/blob/master/image/cat0.png" height="42%" width="42%">　　 <img src="https://github.com/em-1001/YOLOv3/blob/master/image/cat1.png" height="42%" width="42%"></p>

### configuration  
```py
# YOLOv3
OPTIMIZER = Adam
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


# YOLOv4
OPTIMIZER = SGD
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.949
WEIGHT_DECAY = 0.0005
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

ANCHORS = [
    [(0.30, 0.23), (0.41, 0.52), (0.99, 0.87)],
    [(0.07, 0.16), (0.16, 0.11), (0.15, 0.31)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
```

```py
# train.py

import config
import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):

    """reference : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/loss.py"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 1 #0.5
        self.lambda_obj = 1
        self.lambda_box = 1 #2.5

    def forward(self, predictions, target, anchors, box_loss="MSE"):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i


        # No Object Loss
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )


        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce(
          (predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj])
        )


        # Box Loss
        if box_loss == "MSE":
            predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
            target[..., 3:5] = torch.log(
                (1e-16 + target[..., 3:5] / anchors)
            )  # width, height coordinates
            box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        elif box_loss == "IoU" or box_loss == "GIoU" or box_loss == "DIoU" or box_loss == "CIoU":
            anchors = anchors.reshape(1, 3, 1, 1, 2) # 3(anchor) x 2(h, w), p_w * exp(t_w)를 연산하기 위해 reshape
            box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
            iou_loss = 1 - intersection_over_union(boxes_preds=box_preds[obj], boxes_labels=target[..., 1:5][obj], box_format="midpoint", iou_mode = box_loss)
            # box_loss = sum(iou_loss)/len(iou_loss)
            box_loss = iou_loss.mean()



        # Class Loss
        """
        class_target = torch.zeros_like(predictions[..., 5:][obj])
        for i in range(class_target.shape[0]):
            class_target[i][int(target[..., 5][obj][i])] = 1.

        class_loss = self.bce(
            (predictions[..., 5:][obj]), (class_target)
        )"""

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
```

```
yolov3 : 0 -> 5 -> 10 -> 15 -> 15 -> 18 ...
yolov4 : ? -> ? -> 9 -> 15 -> 16 -> 16
```


### NMS
|Detection|320 x 320|416 x 416|512 x 512|
|--|--|--|--|
|YOLOv3|?|31.7|?|
|CSP|?|?|?|
|CSP+GIoU|?|?|?|
|CSP+GIoU+CA|?|?|?|
|CSP+GIoU+CA+Mosaic|?|?|?|

### DIoU-NMS
|Detection|320 x 320|416 x 416|512 x 512|
|--|--|--|--|
|YOLOv3+MSE DIoU-NMS|?|?|?|
|CSP+CIoU+CA DIoU-NMS|?|?|?|

config file에서 image size 바꿔가면서 실험 (320 x 320, 416 x 416, 512 x 512)  
confidence 몇으로 할지 다시생각  
https://csm-kr.tistory.com/62

The model was evaluated with confidence 0.2 and IOU threshold 0.45 using NMS.

# Reference
## Web Link 
One-stage object detection : https://machinethink.net/blog/object-detection/   
DIoU, CIoU : https://hongl.tistory.com/215  
YOLOv3 : https://herbwood.tistory.com/21    
&#160;&#160;&#160;&#160;&#160;&#160;&#160;　　 https://csm-kr.tistory.com/11   
&#160;&#160;&#160;&#160;&#160;&#160;&#160;　　 https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e  
YOLOv4 : https://wikidocs.net/181720   
　　　 　https://csm-kr.tistory.com/62  
Residual block : https://daeun-computer-uneasy.tistory.com/28  
　　　　　　　https://techblog-history-younghunjo1.tistory.com/279     
NMS : https://wikidocs.net/142645     
mAP : https://ctkim.tistory.com/entry/mAPMean-Average-Precision-%EC%A0%95%EB%A6%AC   
BottleNeck : https://velog.io/@lighthouse97/CNN%EC%9D%98-Bottleneck%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4   



## Paper
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf      
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
YOLOv4 : https://arxiv.org/pdf/2004.10934.pdf  
DIoU, CIoU : https://arxiv.org/pdf/1911.08287.pdf    
DenseNet : https://arxiv.org/pdf/1608.06993.pdf    
CSPNet : https://arxiv.org/pdf/1911.11929.pdf    
SPPNet : https://arxiv.org/pdf/1406.4729.pdf    
SGDR : https://arxiv.org/pdf/1608.03983v5.pdf  
