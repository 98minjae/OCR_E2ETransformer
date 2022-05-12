# E2E_OCR_Transformer

*2022 Winter Project*    

**YAI(Yonsei university Artificial Intelligence) with Lomin(https://lomin.ai/)**    

TEAM :  강수빈, 김주연, 변지혁, 이민재, 이상민

---
# Introduction

* 기존한계
  * 기존 OCR E2E model은 detection과 recognition model을 단순 결합하는 방식
  * 이에 training 과정에서 model끼리 서로 영향을 주지 못함 

* 목표
  * detection model과 recognition model이 서로 영향을 줄 수 있는 모델 설계
  * 영어에 비해 구조가 복잡한 한국어에도 잘 작동할 수 있는 모델 설계

* 아이디어
  * 하나의 backbone에서 뽑은 feature map을 detection, recognition model이 공유
  * 4가지 loss 사용
    * label loss
    * bbox loss
    * text loss
    * recog loss

---
# Model

<figure>
<img src="https://user-images.githubusercontent.com/92682815/168018562-3e35f614-311f-49f6-af83-27d24b5c209b.jpg" alt="Trulli" style="width:100%">
</figure> <br/>


**Model Pipeline**

Shared Convolution - FOTS
* Backbone: Resnet-50
* Output: Feature map

Transformer Detection - DETR
* Role: 이미지 내 텍스트 위치 파악
* Output: 이미지 내 Text 존재 유무, Bounding Box 좌표

RoIRotate - FOTS 
* Role: Bounding box 좌표대로 텍스트 원본 이미지에서 자르기
* Role: 이미지 조정 (기울어지거나 보는 관점에서 달라지는 경우)

Transformer Recognition - ViTSTR
* Role: 이미지 내 텍스트 
* Output: Text

---
**Data**

- [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4 "icdar 2015")
- [SynthText_eng](https://rrc.cvc.uab.es/?ch=4 "synthtext eng")
- [SynthText_kor](https://rrc.cvc.uab.es/?ch=4 "synthtext kor")

---
**OCR Transformer Train**

Train Model :
`python _main_.py`

---
# Discussion  

1. Problem

  * DETR 모델 - DETR 모델은 작은 객체를 잘 탐지 못하는 문제점이 있음
  * 해상도 - 메모리로 인한 이미지 해상도 낮추는 한계
  * 예측된 Bounding box 개수 - 이미지 내 gt 텍스트 개수보다 많은 예측된 텍스트 bbox

2. Efforts

 (1) DETR pretrain 하지않기
 - Reason: 기존 DETR 모델이 작은 객체를 탐지못하는 특성 때문
 - Result: 성능 향상 및 gt 텍스트 개수와 예측된 텍스트 개수 일치

 (2) Finetuning ("recog" weight coefficients 높이기)
 - Weight coefficient - class(텍스트 존재 유무) : bounding box : giou : recog
 - Reason: 탐지된 bbox와 인식된 텍스트가 gt와 차이가 너무 난다는 점 때문
 - Result: 효과 없음
    

  (3) Detection 성능 테스트 (+ 해상도 높이기, giou weight coeffients 높이기)
  - Reason: Detection 또는 Recognition task에서 문제가 일어났는지 파악하기 위해
  - Result: 예측되는 bbox개수 감소, giou loss 감소

3. Feedback by LOMIN

* 우리가 만든 모델에서 가장 크게 지적받은 부분은 바로 architecture의 복잡성이었다. 모델의 구조가 복잡해지고, 단계가 많아질수록, 각 모듈들에 대한 검증은 어려워진다. 이러한 부분에서 Lomin측에서는 크게 두가지 부분을 피드백하였다.

  (1) 첫째, 모듈단위에 대한 선검증 후, 전체 모델에 대한 검증으로 이어져야 한다. 단일 Transformer하나만 하더라도 여러개의 component로 이루어져있다. 이러한 transformer가 수없이 쌓인 decoder, 그리고 거기서 이어지는 RoI Rotate, 마지막으로 transformer Encoder까지 우리가 만든 모델은 수많은 component들로 구성되어 있다. 이렇게 큰덩어리의 모듈들로 모델이 구성이 되어 있는 경우, Lomin측에서는 검증을 모듈단위로 먼저 마친 후에 전체 모델에 대한 검증을 하는 것이 유리하다고 피드백을 남겨주었다.

  (2) 둘째, 마찬가지로 모델의 depth가 매우 깊기 때문에 전반적으로 모델이 underfitting되어 있다. Transformer에서 가장 중요한 것은 finetuning과 pre-training이다. 따라서, 로민측에서는 이런 경우 Finetuning을 할때에 각 encoder decoder 별로 먼저 hard-training시킨 후에 training하는 방법을 제안하였다. 예를 들어, 각 encoder decoder 모듈별로 data 1개만 가지고 overtraining 시킨후, 2개, 3개, 5개, 10개 이런식으로 dataset size를 점차적으로 늘려서 overfitting한 후에, 전체적으로 training 시키는 방법을 피드백으로 남겨주었다.


