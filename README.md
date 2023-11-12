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

## Loss

$$λ_ {coord} \sum_ {i=0}^{S^2} \sum_ {j=0}^B 𝟙^{obj}_ {i j} \left[(t_ {x_ i} - \hat{t_ {x_ i}})^2 + (t_ {y_ i} - \hat{t_ {y_ i}})^2 \right]$$

$$+λ_ {coord} \sum_ {i=0}^{S^2} \sum_ {j=0}^B 𝟙^{obj}_ {i j} \left[(t_ {w_ i} - \hat{t_ {w_ i}})^2 + (t_ {h_ i} - \hat{t_ {h_ i}})^2 \right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[-(o_i\log(\hat{o_i}) + (1 - o_i)\log(1 - \hat{o_i}))\right]$$

$$+Mask_{ig} \cdot λ_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{noobj}_{i j} \left[-(o_i\log(\hat{o_i}) + (1 - o_i)\log(1 - \hat{o_i}))\right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[-(c_i\log(\hat{c_i}) + (1 - c_i)\log(1 - \hat{c_i}))\right]$$  



# Cross Entropy
Cross Entropy 는 정보 이론에서 파생된 개념으로, 확률 분포의 차이를 측정하는 지표이다. 두 확률 분포 간의 유사성을 평가하거나, 분류 문제에서 예측값과 실제값 간의 차이를 계산하는 데 사용된다. 

데이터 확률 분포를 $P(x)$, 모델이 추정하는 확률 분포를 $Q(x)$라고 할 때, 두 확률 분포 $P$와 $Q$의 차이를 측정하는 지표인 Cross Entropy는 아래와 같이 표현된다. 

$$H(p, q) = H(p) + D_{KL}(p||q) = -\sum_{i=0}^{n} p(x_i)\log{(q(x_i))}$$

일반적인 Classification 문제에서 주로 cross entropy loss를 사용한다. 이때 True distribution $P$는 one-hot 인코딩된 vector(Ground Truth)를 사용한다. Prediction distribution $Q$ 는 모델의 예측 값으로 softmax layer를 거친 후의 값이고, 클래스 별 확률 값을 모두 합치면 1이 된다. 

예를 들어 $P = [0, 1, 0]$,  $Q = [0.2, 0.7, 0.1]$ 일 때, cross entropy loss 결과는 아래와 같다.

$$
\begin{aligned}
H(P,Q)&=-\sum P(x)log Q(x)\\
&=-(0 \cdot \log{0.2} + 1 \cdot \log{0.7} + 0 \cdot \log{0.1})\\
&=-\log{0.7}
\end{aligned}$$

이진 분류 문제에서의 cross entropy는 다음과 같이 표현된다. 

$$H(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N (y_i \log{(\hat{y}_i)} + (1-y_i) \log{(1 - \hat{y}_i)})$$

$y_i$는 실제 클래스를 나타내는 값(0 또는 1)이고, $\hat{y}_i$는 모델의 예측 확률을 나타낸다.   

위 식은 다음과 같이 유도된다.
1. 정보 이론적 관점  
이진 분류 문제에서, 실제값 $y$가 1이라면 모델은 1로 예측해야 하며, cross entropy는 $-log(\hat{y})$가 되야한다.
반대로 실제값 $y$가 0이라면 모델은 0으로 예측해야 하며, cross entropy는 $-log(1 - \hat{y})$가 된다.
2. 평균화 및 합산  
이러한 관점에서 전체 데이터셋에 대해 각각의 경우(실제값이 1 또는 0인 경우)의 교차 엔트로피를 합산하여 평균을 취한 것이 최종적인 이진 분류 문제의 교차 엔트로피 손실 함수의 식이 된다.


위 식은 실제 분포(Ground Truth)인 $y_i$와 모델의 예측인 $\hat{y}_i$사이의 정보량을 측정하고 모델이 Ground Truth와 일치할수록, Cross Entropy의 값은 작아진다.  

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

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4abe5f78-388b-459f-a3f4-95e41a5fdb0a" height="30%" width="30%"></p>

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


## Paper
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf      
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
DIoU, CIoU : https://arxiv.org/pdf/1911.08287.pdf
