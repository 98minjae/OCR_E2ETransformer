# E2E_OCR_Transformer

*2022 Winter Project*    

**YAI(Yonsei univ) with Lomin**    

Contributors : 강수빈, 김주연, 변지혁, 이민재, 이상민

---
**Model**



<figure>
<img src="https://user-images.githubusercontent.com/92682815/168018562-3e35f614-311f-49f6-af83-27d24b5c209b.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>MODEL pipeline</b></figcaption>
</figure> <br/>



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
**Discussion**  

1. Problem

  * DETR 모델 - DETR 모델은 작은 객체를 잘 탐지 못하는 문제점이 있음
  * 해상도 - 메모리로 인한 이미지 해상도 낮추는 한계
  * 예측된 Bounding box 개수 - 이미지 내 gt 텍스트 개수보다 많은 예측된 텍스트 bbox

2. Efforts

 (1) DETR pretrain 하지않기
 - Reason: 기존 DETR 모델이 작은 객체를 탐지못하는 특성 때문
 - Result: 성능 향상 및 gt 텍스트 개수와 예측된 텍스트 개수 일치

<img width="269" alt="그림1" src="https://user-images.githubusercontent.com/50818298/168034411-19ad2a9e-8c85-44b8-9175-54620eaf390d.png"> Previous vs X Pretrained 

<img width="289" alt="pretrained" src="https://user-images.githubusercontent.com/50818298/168034701-9efee334-af79-4f6c-9be3-270af90a4af9.png"> Pretrained
<img width="289" alt="raw" src="https://user-images.githubusercontent.com/50818298/168034723-ec4cff40-1cd1-423a-bdb6-07ae29af5efa.png"> X Pretrained
 
 (2) Finetuning ("recog" weight coefficients 높이기)
 - Weight coefficient - class(텍스트 존재 유무) : bounding box : giou : recog
 - Reason: 탐지된 bbox와 인식된 텍스트가 gt와 차이가 너무 난다는 점 때문
 - Result: 효과 없음
    
<img width="255" alt="weight_coefficient" src="https://user-images.githubusercontent.com/50818298/168037360-7d28b008-7e24-4382-8e5d-b9d5c8bfb943.png"> Weight Coefficient 1:5:2:2 vs 1:5:2:4

<img width="290" alt="1-5-2-2" src="https://user-images.githubusercontent.com/50818298/168037605-be880f0b-3fb1-4a7a-a5c2-4037ed4280f5.png"> 1:5:2:2
<img width="291" alt="1-5-2-4" src="https://user-images.githubusercontent.com/50818298/168037620-ce62f9f9-4879-40e6-a524-ce16300a2b70.png"> 1:5:2:4
 

  (3) Detection 성능 테스트 (+ 해상도 높이기, giou weight coeffients 높이기)
  - Reason: Detection 또는 Recognition task에서 문제가 일어났는지 파악하기 위해
  - Result: 예측되는 bbox개수 감소, giou loss 감소

<img width="197" alt="only_detection" src="https://user-images.githubusercontent.com/50818298/168038084-591d5d33-ce72-472b-aebb-0012d929b3e6.png"> Only Detection vs Previous
![only_detection_1](https://user-images.githubusercontent.com/50818298/168038104-e9ed7ea1-6bc1-438f-8fb7-9e6c6b51b50d.png) Only Detection

3. Feedback

* 또 뭐있지?
