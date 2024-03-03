# YOLOv4
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
