# E2E_OCR_Transformer

*2022 Winter Project*    

**[YAI](https://yai-yonsei.tistory.com/)(Yonsei university Artificial Intelligence) with [Lomin](https://lomin.ai/)**    

TEAM : Subin Kang(강수빈), Juyeon Kim(김주연), JiHyuk Byun(변지혁), MinJae Lee(이민재), Sangmin Lee(이상민)


---
# Introduction  

Many OCR models are either a detection model or a recognition model. In order words, the two tasks are accomplished through separate models. So, there are times in which detection and recognition aren't incompatible, while training. Our goal is to make detecting and recognizing texts within images through a single step with transformer. Also, we tried to recognize Korean texts as well as English ones. Our main idea is to share the features extracted by backbone(Resnet50) with the detection branch and the recognition branch, inspired by FOTS, and to use four losses (label loss, bbox loss, text loss, recognition loss).


---
# Model

<figure>
<img src="https://user-images.githubusercontent.com/92682815/168018562-3e35f614-311f-49f6-af83-27d24b5c209b.jpg" alt="Trulli" style="width:100%">
</figure> <br/>


**Model Pipeline**

(1) **Shared Convolution (from FOTS)**
* Backbone: Resnet-50
* Extracting the features
* Output: Feature map

(2) **Transformer Detection from DETR**
* Detecting the texts in images
* Output: binary predicttion of presence(1) or absence(0) of texts in image queries and Bounding Box coordinates

(3) **RoIRotate from FOTS** 
* Cropping the images with Bounding box coordinates
* Rotating the region of interests to recognize the texts easily

(4) **Transformer Recognition from ViTSTR**
* Recognizing the texts in the images 
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
# OCR Transformer Train

Train Model :
`python _main_.py`

---
# Discussion  

* Problem
  - DETR Model – The DETR model isn’t suitable for detecting small-sized objects (texts)
  - Resolution – Limitation in lowering image resolution due to memory
  - Number of predicted bounding boxes - # of predicted text bbox > # of gt text bbox

* Trial and errors  

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

* Feedback by Lomin  

   **The architecture complexity was the most signified problem pointed out.** 

   - Prior to validating the entire model, validations of each module should have been processed. Our model consists of countless components. When such many, large-sized modules are utilized, it would be productive to validate each module step by step.

   - Our model is too deep, the model undergoes underfitting as a whole. Lomin advised our team to first hard-train the encoder and decoder separately when finetuning, before training. For instance, we could overfit the model by training each encoder and decoder with just a single data, consecutively increase the dataset size, and finally train our entire dataset.

For details of our project,
go to [YAI & LOMIN COPORATE PROJECT](https://lively-silence-df6.notion.site/E2E_OCR_Transformer-34767a7a0a1641899e241b3fd210f2b9)
