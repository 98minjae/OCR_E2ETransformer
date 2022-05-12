# E2E_OCR_Transformer

*2022 Winter Project*    

**YAI([Yonsei university Artificial Intelligence](https://yai-yonsei.tistory.com/)) with [Lomin](https://lomin.ai/)**    

TEAM :  강수빈, 김주연, 변지혁, 이민재, 이상민


---
# Introduction  

Many OCR models are either a detection model or a recognition model. In order words, the two tasks are accomplished through separate models. So, there are times in which detection and recognition aren't incompatible, while training. Our goal is to make detecting and recognizing texts within images through a single step with transformer. Also, we tried to recognize Korean texts as well as English ones. Our main idea is to share the features extracted by backbone(Resnet50) with the detection branch and the recognition branch, inspired by FOTS, and to use four losses (label loss, bbox loss, text loss, recognition loss).


---
# Model

<figure>
<img src="https://user-images.githubusercontent.com/92682815/168018562-3e35f614-311f-49f6-af83-27d24b5c209b.jpg" alt="Trulli" style="width:100%">
</figure> <br/>


**Model Pipeline**

(1) Shared Convolution (from FOTS)
* Backbone: Resnet-50
* Output: Feature map

(2) Transformer Detection from DETR
* Role: 이미지 내 텍스트 위치 파악
* Output: 이미지 내 Text 존재 유무, Bounding Box 좌표

(3) RoIRotate from FOTS 
* Role: Bounding box 좌표대로 텍스트 원본 이미지에서 자르기
* Role: 이미지 조정 (기울어지거나 보는 관점에서 달라지는 경우)

(4 )Transformer Recognition from ViTSTR
* Role: 이미지 내 텍스트 
* Output: Text

---
# Structure

<pre>
<code>datasets
   __init__.py
   coco_eval.py
   text.py
   transforms.py

util            # roi_rotate.py is hevily based on FOTS    
   __init__.py      # Here is Repo: https://github.com/jiangxiluning/FOTS.PyTorch/tree/master/FOTS/model/modules (FOTS)
   box_ops.py
   misc.py
   plot_utils.py
   upsampling.py
   visualize_results.py
   convert.py
   matcher.py
   metric.py
   position_encoding.py
   roi_rotate.py

models         # Heavily based on DETR and ViTSTR
            # Here is Repo : https://github.com/facebookresearch/detr (DETR)
                      https://github.com/roatienza/deep-text-recognition-benchmark (ViTSTR)   
   model.py      
   backbone.py         
       loss.py                      
   transformer.py
   vitstr.py

_main_.py
_train_.py
config.py

</code>
</pre>
---
# Data

- [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4 "icdar 2015")
- [SynthText_eng](https://rrc.cvc.uab.es/?ch=4 "synthtext eng")
- [SynthText_kor](https://rrc.cvc.uab.es/?ch=4 "synthtext kor")

---
# OCR Transformer Train**

Train Model :
`python _main_.py`

---
# Discussion  

* Problem
  - DETR Model – The DETR model isn’t suitable for detecting small-sized objects (texts)
  - Resolution – Limitation in lowering image resolution due to memory
  - Number of predicted bounding boxes - # of predicted text bbox > # of gt text bbox

* Efforts  

  (1) Omit pretraining DETR
  - Reason: Existing DETR model shows low performance in small-sized objects
  - Result: Improved performance and Accordance of # of predicted text bbox and # of gt text bbox

  (2) Finetuning (Increase weight coefficient of “recog” loss)
  - Weight coefficient – class(True/False) : bounding box : giou : recog
  - Reason: Deviation of recognized text from gt text
  - Result: No effect

  (3) Detection performance test (+ Increase resolution and giou loss weight coefficient)
  - Reason: To identity whether the detection or the recognition task was the problem
  - Result: Decrease in number of predicted text bbox and giou loss

* Feedback by LOMIN  
  우리가 만든 모델에서 가장 크게 지적받은 부분은 바로 architecture의 복잡성이었다. 모델의 구조가 복잡해지고, 단계가 많아질수록, 각 모듈들에 대한 검증은 어려워진다. 이러한 부분에서 Lomin측에서는 크게 두가지 부분을 피드백하였다.

   (1) 모듈단위에 대한 선검증 후, 전체 모델에 대한 검증으로 이어져야 한다. 단일 Transformer하나만 하더라도 여러개의 component로 이루어져있다. 이러한 transformer가 수없이 쌓인 decoder, 그리고 거기서 이어지는 RoI Rotate, 마지막으로 transformer Encoder까지 우리가 만든 모델은 수많은 component들로 구성되어 있다. 이렇게 큰덩어리의 모듈들로 모델이 구성이 되어 있는 경우, Lomin측에서는 검증을 모듈단위로 먼저 마친 후에 전체 모델에 대한 검증을 하는 것이 유리하다고 피드백을 남겨주었다.

   (2) 마찬가지로 모델의 depth가 매우 깊기 때문에 전반적으로 모델이 underfitting되어 있다. Transformer에서 가장 중요한 것은 finetuning과 pre-training이다. 따라서, 로민측에서는 이런 경우 Finetuning을 할때에 각 encoder decoder 별로 먼저 hard-training시킨 후에 training하는 방법을 제안하였다. 예를 들어, 각 encoder decoder 모듈별로 data 1개만 가지고 overtraining 시킨후, 2개, 3개, 5개, 10개 이런식으로 dataset size를 점차적으로 늘려서 overfitting한 후에, 전체적으로 training 시키는 방법을 피드백으로 남겨주었다.


