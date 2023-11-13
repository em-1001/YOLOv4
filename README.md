# YOLOv3
## Bounding Box
<p align="center"><img src="https://github.com/em-1001/YOLOv3-CIoU/assets/80628552/b7058b48-1120-409e-ae7c-1c5ab8b09159">






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
<p align="center"><img src="https://github.com/em-1001/YOLOv3-CIoU/assets/80628552/580c083b-db62-47bf-af15-95396efb576c" height="40%" width="40%"></p>

## Loss

$$λ_ {coord} \sum_ {i=0}^{S^2} \sum_ {j=0}^B 𝟙^{obj}_ {i j} \left[(t_ {x_ i} - \hat{t_ {x_ i}})^2 + (t_ {y_ i} - \hat{t_ {y_ i}})^2 \right]$$

$$+λ_ {coord} \sum_ {i=0}^{S^2} \sum_ {j=0}^B 𝟙^{obj}_ {i j} \left[(t_ {w_ i} - \hat{t_ {w_ i}})^2 + (t_ {h_ i} - \hat{t_ {h_ i}})^2 \right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[-(o_i\log(\hat{o_i}) + (1 - o_i)\log(1 - \hat{o_i}))\right]$$

$$+Mask_{ig} \cdot λ_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{noobj}_{i j} \left[-(o_i\log(\hat{o_i}) + (1 - o_i)\log(1 - \hat{o_i}))\right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[-(c_i\log(\hat{c_i}) + (1 - c_i)\log(1 - \hat{c_i}))\right]$$  

$S$ : number of cells    
$B$ : number of anchors  
$o$ : objectness  
$c$ : class label  
$λ_ {coord}$ : coordinate loss balance constant  
$λ_{noobj}$ : no confidence loss balance constant    
$𝟙^{obj}_ {i j}$ : 1 when there is object, 0 when there is no object  
$𝟙^{noobj}_ {i j}$ : 1 when there is no object, 0 when there is object  
$Mask_{ig}$ : tensor that masks only the anchor with iou $\le$ 0.5. Have a shape of $\left[S, S, B\right]$.

$o$ (objectness)는 anchor와 bbox의 iou가 가장 큰 anchor의 값이 1, 그렇지 않은 경우의 값이 0인 $\left[13, 13, 3, 1\right]$의 tensor로 만들어진다. $c$ (class label)은 one-encoding으로 $\left[13, 13, 3, n \right]$ ($n$ : num_classes) 의 shape를 갖는 tensor로 만들어진다. 


# GIoU, DIoU,  CIoU
일반적으로 IoU-based loss는 다음과 같이 표현된다. 

$$L = 1 - IoU + \mathcal{R}(B, B^{gt})$$

여기서 $R(B, B^{gt})$는  predicted box $B$와 target box $B^{gt}$에 대한 penalty term이다.  
$1 - IoU$로만 Loss를 구할 경우 box가 겹치지 않는 case에 대해서 어느 정도의 오차로 교집합이 생기지 않은 것인지 알 수 없어서 gradient vanishing 문제가 발생했다. 이러한 문제를 해결하기 위해 penalty term을 추가한 것이다. 

## Generalized-IoU(GIoU)
Generalized-IoU(GIoU) 의 경우 Loss는 다음과 같이 계산된다. 

$$L_{GIoU} = 1 - IoU + \frac{|C - B ∪ B^{gt}|}{|C|}$$

여기서 $C$는 $B$와 $B^{gt}$를 모두 포함하는 최소 크기의 Box를 의미한다. Generalized-IoU는 겹치지 않는 박스에 대한 gradient vanishing 문제는 개선했지만 horizontal과 vertical에 대해서 에러가 크다. 이는 target box와 수평, 수직선을 이루는 Anchor box에 대해서는 $|C - B ∪ B^{gt}|$가 매우 작거나 0에 가까워서 IoU와 비슷하게 동작하기 때문이다. 또한 겹치지 않는 box에 대해서 일단 predicted box의 크기를 매우 키우고 IoU를 늘리는 동작 특성 때문에 수렴 속도가 매우 느리다. 

## Distance-IoU(DIoU)
GIoU가 면적 기반의 penalty term을 부여했다면, DIoU는 거리 기반의 penalty term을 부여한다. 
DIoU의 penalty term은 다음과 같다. 

$$\mathcal{R}_{DIoU} = \frac{\rho^2(b, b^{gt})}{c^2}$$

$\rho^2$는 Euclidean거리이며 $c$는 $B$와 $B^{gt}$를 포함하는 가장 작은 Box의 대각선 거리이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4abe5f78-388b-459f-a3f4-95e41a5fdb0a" height="25%" width="25%"></p>

DIoU Loss는 두 개의 box가 완벽히 일치하면 0, 매우 멀어지면 $L_{GIoU} = L_{DIoU} \mapsto 2$가 된다. 이는 IoU가 0이 되고, penalty term이 1에 가깝게 되기 때문이다. Distance-IoU는 두 box의 중심 거리를 직접적으로 줄이기 때문에 GIoU에 비해 수렴이 빠르고, 거리기반이므로 수평, 수직방향에서 또한 수렴이 빠르다. 

## Complete-IoU(CIoU)
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

# Reference
## Web Link 
One-stage object detection : https://machinethink.net/blog/object-detection/   
DIoU, CIoU : https://hongl.tistory.com/215  
YOLOv3 : https://herbwood.tistory.com/21  
&#160;&#160;&#160;&#160;&#160;　　　 https://csm-kr.tistory.com/11


## Paper
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf      
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
DIoU, CIoU : https://arxiv.org/pdf/1911.08287.pdf
