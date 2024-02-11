# Performance
<img src="https://github.com/em-1001/YOLOv3/blob/master/image/cat0_1.png">&#160;&#160;&#160;&#160;<img src="https://github.com/em-1001/YOLOv3/blob/master/image/cat1_1.png"> 

### Configuration  
```ini
DATASET = PASCAL_VOC
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 

BATCH_SIZE = 32
OPTIMIZER = Adam
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
WEIGHT_DECAY = 1e-4

# 0 ~ 30 epoch                # Cosine Annealing                            

LEARNING_RATE = 0.0001        LEARNING_RATE = 0.0001        
                              T_max=100
# 30 ~ 50 epoch               

LEARNING_RATE = 0.00005       

# 50 ~  epoch                

LEARNING_RATE = 0.00001      

```

### NMS(Non-maximum Suppression)
|Detection|320 x 320|416 x 416|512 x 512|
|--|--|--|--|
|CSP|?|43.5|?|
|CSP + GIoU|?|?|?|
|CSP + DIoU|?|?|?|
|CSP + CIoU|?|46.4|?|
|CSP + SIoU|?|46.2|?|
|CSP + CIoU + CA|?|?|?|
|CSP + CIoU + CA + M|?|?|?|

### DIoU-NMS
|Detection|320 x 320|416 x 416|512 x 512|
|--|--|--|--|
|CSP + CIoU|?|46.4|?|
|CSP + CIoU + CA|?|?|?| 


# IoU Loss 
일반적으로 IoU-based loss는 다음과 같이 표현된다. 

$$L = 1 - IoU + \mathcal{R}(B, B^{gt})$$

여기서 $R(B, B^{gt})$는  predicted box $B$와 target box $B^{gt}$에 대한 penalty term이다.  
$1 - IoU$로만 Loss를 구할 경우 box가 겹치지 않는 case에 대해서 어느 정도의 오차로 교집합이 생기지 않은 것인지 알 수 없어서 gradient vanishing 문제가 발생했다. 이러한 문제를 해결하기 위해 penalty term을 추가한 것이다. 
## Generalized-IoU(GIoU)
Generalized-IoU(GIoU) 의 경우 Loss는 다음과 같이 계산된다. 

$$L_{GIoU} = 1 - IoU + \frac{|C - B ∪ B^{gt}|}{|C|}$$

여기서 $C$는 $B$와 $B^{gt}$를 모두 포함하는 최소 크기의 Box를 의미한다. Generalized-IoU는 겹치지 않는 박스에 대한 gradient vanishing 문제는 개선했지만 horizontal과 vertical에 대해서 에러가 크다. 이는 target box와 수평, 수직선을 이루는 Anchor box에 대해서는 $|C - B ∪ B^{gt}|$가 매우 작거나 0에 가까워서 IoU와 비슷하게 동작하기 때문이다. 또한 겹치지 않는 box에 대해서 일단 predicted box의 크기를 매우 키우고 IoU를 늘리는 동작 특성 때문에 수렴 속도가 느리다. 

## Distance-IoU(DIoU)
GIoU가 면적 기반의 penalty term을 부여했다면, DIoU는 거리 기반의 penalty term을 부여한다. 
DIoU의 penalty term은 다음과 같다. 

$$\mathcal{R}_{DIoU} = \frac{\rho^2(b, b^{gt})}{c^2}$$

$\rho^2$는 Euclidean거리이며 $c$는 $B$와 $B^{gt}$를 포함하는 가장 작은 Box의 대각선 거리이다. 

DIoU Loss는 두 개의 box가 완벽히 일치하면 0, 매우 멀어지면 $L_{GIoU} = L_{DIoU} \to 2$가 된다. 이는 IoU가 0이 되고, penalty term이 1에 가깝게 되기 때문이다. Distance-IoU는 두 box의 중심 거리를 직접적으로 줄이기 때문에 GIoU에 비해 수렴이 빠르고, 거리기반이므로 수평, 수직방향에서 또한 수렴이 빠르다. 

### DIoU-NMS
DIoU를 NMS(Non-Maximum Suppression)에도 적용할 수 있다. 일반적인 NMS의 경우 이미지에서 같은 class인 두 물체가 겹쳐있는 Occlusion(가림)이 발생한 경우 올바른 박스가 삭제되는 문제가 발생하는데, DIoU를 접목할 경우 두 박스의 중심점 거리도 고려하기 때문에 target box끼리 겹쳐진 경우에도 robust하게 동작할 수 있다. 

$$
s_i =
\begin{cases}
s_ i, & IoU - \mathcal{R}_ {DIoU}(\mathcal{M}, B_i) < \epsilon\\
0, & IoU - \mathcal{R}_{DIoU}(\mathcal{M}, B_i) \ge \epsilon
\end{cases}
$$

가장 높은 Confidence score를 갖는 $\mathcal{M}$에 대해 IoU와 DIoU의 distance penalty를 동시에 고려하여 IoU가 매우 크더라도 중심점 사이의 거리가 멀면 다른 객체를 탐지한 것일 수도 있으므로 위와 같이 일정 임계치 $\epsilon$ 보다 작으면 없애지 않고 보존한다. 


## Complete-IoU(CIoU)
CIoU에서는 **overlap area**, **central point distance**, **aspect ratio**를 고려한다. 이 중 overlap area, central point는 DIoU에서 이미 다뤘고 여기에 aspect ratio에 대한 penalty term을 추가한 것이 CIoU이다. CIoU penalty term는 다음과 같이 정의된다. 

$$\mathcal{R}_{CIoU} = \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

$$v = \frac{4}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}})^2$$

$$\alpha = \frac{v}{(1 - IoU) + v}$$

$v$의 경우 bbox는 직사각형이고 $\arctan{\frac{w}{h}} = \theta$이므로 $\theta$의 차이를 통해 aspect ratio를 구하게 된다. 이때 $v$에 $\frac{2}{π}$가 곱해지는 이유는 $\arctan$ 함수의 최대치가 $\frac{2}{π}$ 이므로 scale을 조정해주기 위해서이다. 

$\alpha$는 trade-off 파라미터로 IoU가 큰 box에 대해 더 큰 penalty를 주게 된다. 

CIoU에 대해 최적화를 수행하면 아래와 같은 기울기를 얻게 된다. 이때, $w, h$는 모두 0과 1사이로 값이 작아 gradient explosion을 유발할 수 있다. 따라서 실제 구현 시에는 $\frac{1}{w^2 + h^2} = 1$로 설정한다. 

$$\frac{\partial v}{\partial w} = \frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{h}{w^2 + h^2}$$ 

$$\frac{\partial v}{\partial h} = -\frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{w}{w^2 + h^2}$$ 

## SCYLLA-IoU(SIoU)
SCYLLA-IoU(SIoU)는 **Angle cost**, **Distance cost**, **Shape cost**를 고려하며 penalty term은 다음과 같다.

$$\mathcal{R}_{SIoU} = \frac{\Delta + \Omega}{2}$$

### Angle cost
Angle cost는 다음과 같이 계산된다. 

$$\begin{align}
\Lambda &= 1 - 2 \cdot \sin^2\left(\arcsin(x) - \frac{\pi}{4} \right) \\   
&= 1 - 2 \cdot \sin^2\left(\arcsin(\sin(\alpha)) - \frac{\pi}{4} \right) \\
&= 1 - 2 \cdot \sin^2\left(\alpha - \frac{\pi}{4} \right) \\
&= \cos^2\left(\alpha - \frac{\pi}{4}\right) - \sin^2\left(\alpha - \frac{\pi}{4}\right) \\ 
&= \cos\left(2\alpha - \frac{\pi}{2}\right) \\ 
&= \sin(2\alpha) \\ 
\end{align}$$

$$\begin{align}
&\\ 
&where \\ 
&\\  
&x = \frac{c_h}{\sigma} = \sin(\alpha) \\  
&\sigma = \sqrt{(b_{c_x}^{gt} - b_{c_x})^2 + (b_{c_y}^{gt} - b_{c_y})^2} \\  
&c_h = \max(b_{c_y}^{gt}, b_{c_y}) - \min(b_{c_y}^{gt}, b_{c_y})
\end{align}$$

만약 $\alpha > \frac{\pi}{4}$라면 $\beta = \frac{\pi}{2} - \alpha$로 바꿔서 베타로 계산한다. 

### Distance cost 
Distance cost에 Angle cost가 포함되며 다음과 같이 계산된다. 

$$\begin{align}
&\Delta = \sum_{t=x,y} (1 - e^{-\gamma \rho_t}) \\ 
&\\ 
&where \\ 
&\\  
&\rho_ x = \left(\frac{b_{c_x}^{gt} - b_{c_x}}{c_w} \right)^2, \ \rho_ y = \left(\frac{b_{c_y}^{gt} - b_{c_y}}{c_h} \right)^2, \ \gamma = 2 - \Lambda
\end{align}$$

여기서의 $c_w, c_h$는 Angle cost와는 달리 $B$와 $B^{gt}$를 포함하는 가장 작은 Box의 width와 height이다.   
Distance cost를 보면 $\alpha \to 0$일 때 급격하게 작아지고, $\alpha \to \frac{\pi}{4}$일 때 커지기 때문에, $\gamma$가 이를 조정해주는 역할을 한다. 

### Shape cost
Shape cost는 다음과 같이 계산된다. 

$$\begin{align}
&\Omega = \sum_{t=w,h} (1-e^{-\omega_t})^{\theta} \\ 
&\\ 
&where \\ 
&\\  
&\omega_w = \frac{|w-w^{gt}|}{\max(w,w^{gt})}, \omega_h = \frac{|h-h^{gt}|}{\max(h,h^{gt})} \\   
\end{align}$$

$\theta$는 Shape cost에 얼마의 비중을 둘 지 정하며, 보통 4로 설정하고 2에서 6사이의 값으로 한다. 

최종적인 Loss는 다음과 같다. 

$$L_{SIoU} = 1 - IoU + \frac{\Delta + \Omega}{2}$$

# Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos{\left(\frac{T_{cur}}{T_{\max}}\pi\right)} \right), \ T_{cur} \neq (2k+1)T_{\max}$$

$\eta_{\min}$ : min learning rate    
$\eta_{\max}$ : max learning rate    
$T_{\max}$ : period

# Pretrained Weights
CSP : https://www.kaggle.com/datasets/sj2129tommy/csp100epochs   
CSP + GIoU :  
CSP + DIoU :  
CSP + CIoU : https://www.kaggle.com/datasets/sj2129tommy/csp-ciou-100epoch          
CSP + SIoU : https://www.kaggle.com/datasets/sj2129tommy/csp-siou100epoch  
CSP + CIoU + CA :   

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
Cosine Annealing : https://ai4nlp.tistory.com/16  


## Paper
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf      
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
YOLOv4 : https://arxiv.org/pdf/2004.10934.pdf  
DIoU, CIoU : https://arxiv.org/pdf/1911.08287.pdf      
SIoU : https://arxiv.org/ftp/arxiv/papers/2205/2205.12740.pdf  
DenseNet : https://arxiv.org/pdf/1608.06993.pdf    
CSPNet : https://arxiv.org/pdf/1911.11929.pdf    
SPPNet : https://arxiv.org/pdf/1406.4729.pdf    
SGDR : https://arxiv.org/pdf/1608.03983v5.pdf  
