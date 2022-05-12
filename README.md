# E2E_OCR_Transformer

*2022 Winter Project*    

**YAI(Yonsei univ) with Lomin**    

Contributors : 강수빈, 김주연, 변지혁, 이민재, 이상민

---
**Model**


<figure>
<img src="https://user-images.githubusercontent.com/92682815/168018562-3e35f614-311f-49f6-af83-27d24b5c209b.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>MODEL pipeline</b></figcaption>
</figure>


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

Icdar 2015, Syntext, Korean,...

---
**Discussion**  
* Problem
  1. Complexity
  2. asd
  3. bce

  
* Feedback
  1. Overfitting while fine-tuning
  2. asdf

* 또 뭐있지?
