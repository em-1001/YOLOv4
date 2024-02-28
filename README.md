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
In general, IoU-based loss is expressed as follows.

$$L = 1 - IoU + \mathcal{R}(B, B^{gt})$$

where $R(B, B^{gt})$ is the penalty term for the predicted box $B$ and the target box $B^{gt}$.  
If the loss is calculated only by $1 - IoU$, it is not possible to know with what loss the intersection has not occurred for the case where the boxes do not overlap, which leads to the problem of gradient vanishing. We added the penalty term to solve this problem. 
## Generalized-IoU(GIoU)
For Generalized-IoU (GIoU), the loss is calculated as follows

$$\mathcal{R}_{GIoU} = \frac{|C - B ∪ B^{gt}|}{|C|}$$

where $C$ is the smallest box that contains both $B$ and $B^{gt}$. Generalized-IoU improves the gradient vanishing problem for non-overlapping boxes, but has large errors for horizontal and vertical. This is because $|C - B ∪ B^{gt}|$ is very small or close to zero for anchor boxes that form a horizontal and vertical line with the target box, so it behaves similarly to IoU. It also has a slow convergence rate due to the behavior of increasing the size of the predicted box very large in order to increase the IoU for non-overlapping boxes.

## Distance-IoU(DIoU)
If the GIoU assigned an area-based penalty term, the DIoU assigns a distance-based penalty term. 
The penalty terms for DIoU are as follows

$$\mathcal{R}_{DIoU} = \frac{\rho^2(b, b^{gt})}{c^2}$$

$\rho^2$ is the Euclidean distance and $c$ is the diagonal distance of the smallest Box containing $B$ and $B^{gt}$.

DIoU Loss is zero when the two boxes are perfectly aligned, and $L_{GIoU} = L_{DIoU} \to 2$ when they are very far apart. This is because the IoU goes to zero and the penalty term approaches 1. Distance-IoU converges faster than GIoU because it directly reduces the center distance of the two boxes, and because it is distance-based, it also converges faster in the horizontal and vertical directions.

### DIoU-NMS
DIoU can also be applied to Non-Maximum Suppression (NMS). With conventional NMS, the correct box is deleted in the case of occlusions where two objects of the same class in the image overlap, but with DIoU, the distance between the center points of the two boxes is also taken into account, so it can behave robustly even when the target boxes overlap.

$$
s_i =
\begin{cases}
s_ i, & IoU - \mathcal{R}_ {DIoU}(\mathcal{M}, B_i) < \epsilon\\
0, & IoU - \mathcal{R}_{DIoU}(\mathcal{M}, B_i) \ge \epsilon
\end{cases}
$$

The DIoU NMS considers the distance penalty of IoU and DIoU simultaneously for the $\mathcal{M}$ with the highest confidence score, so even if the IoU is very large, if the distance between the center points is large, it may have detected another object, so it keeps it instead of discarding it if it is smaller than a certain threshold $\epsilon$ as above.


## Complete-IoU(CIoU)
CIoU considers **overlap area**, **central point distance**, and **aspect ratio**. Of these, overlap area and central point are already covered in DIoU, and CIoU adds a penalty term for aspect ratio. The CIoU penalty term is defined as follows.

$$\mathcal{R}_{CIoU} = \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

$$v = \frac{4}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}})^2$$

$$\alpha = \frac{v}{(1 - IoU) + v}$$

In the case of $v$, the bbox is a rectangle and $\arctan{\frac{w}{h}} = \theta$, so the aspect ratio is obtained through the difference between $\theta$. the reason $v$ is multiplied by $\frac{2}{π}$ is to adjust the scale because the maximum value of the $\arctan$ function is $\frac{2}{π}$.

The $\alpha$ is a trade-off parameter, which penalizes larger boxes with larger IoUs.

When optimizing for CIoU, we get the following gradient. Here, $w, h$ are both small values between 0 and 1, which can cause a gradient explosion. Therefore, we set $\frac{1}{w^2 + h^2} = 1$ in the actual implementation.

$$\frac{\partial v}{\partial w} = \frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{h}{w^2 + h^2}$$ 

$$\frac{\partial v}{\partial h} = -\frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{w}{w^2 + h^2}$$ 

## SCYLLA-IoU(SIoU)
SCYLLA-IoU (SIoU) considers **Angle cost**, **Distance cost**, **Shape cost** and the penalty term is as follows.

$$\mathcal{R}_{SIoU} = \frac{\Delta + \Omega}{2}$$

### Angle cost
Angle cost is calculated as follows

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

If $\alpha > \frac{\pi}{4}$, then $\beta = \frac{\pi}{2} - \alpha$, which is calculated as beta.

### Distance cost 
Distance cost includes Angle cost, which is calculated as follows

$$\begin{align}
&\Delta = \sum_{t=x,y} (1 - e^{-\gamma \rho_t}) \\ 
&\\ 
&where \\ 
&\\  
&\rho_ x = \left(\frac{b_{c_x}^{gt} - b_{c_x}}{c_w} \right)^2, \ \rho_ y = \left(\frac{b_{c_y}^{gt} - b_{c_y}}{c_h} \right)^2, \ \gamma = 2 - \Lambda
\end{align}$$

Here, $c_w, c_h$ are the width and height of the smallest box containing $B$ and $B^{gt}$, unlike the Angle cost.   
If we look at the Distance cost, we can see that it gets sharply smaller as $\alpha \to 0$ and larger as $\alpha \to \frac{\pi}{4}$, so $\gamma$ is there to adjust it.

### Shape cost
Shape cost is calculated as follows

$$\begin{align}
&\Omega = \sum_{t=w,h} (1-e^{-\omega_t})^{\theta} \\ 
&\\ 
&where \\ 
&\\  
&\omega_w = \frac{|w-w^{gt}|}{\max(w,w^{gt})}, \omega_h = \frac{|h-h^{gt}|}{\max(h,h^{gt})} \\   
\end{align}$$

The $\theta$ specifies how much weight to give to the Shape cost, usually set to 4 and can be a value between 2 and 6.

The final loss is

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
