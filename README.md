<center><h2><strong>Dennies Bor</strong><h2></center >
<center><h2><strong>Team Reef_safe</strong><h2></center >
<strong><h3>DATA PREP && MODEL TRAINING<h3></strong >
---
<em>CONCEPT:</em > To (!attempt) build and deploy a realtime coffee/liquid spill warning system using conv net object detectors.
* Coco datasets consists of 90 classes of items. Keyboard, laptop, and mouse inclusive. Familiar with env.
* Take images around my workstation.
* Manually label risks the i.e, mug, bottle of water,etc
* Train a detector(classifier & regressor) to isolate risks.
* Deploy to cloud - cam as a stdin
---
<em />ATTEMPT:</em >
* Collect images with my phone - 60 imgs with and without hazards for labelling.
<br />
<img src="https://github.com/denniesbor/DSI_Twist_Chellenge/blob/assets/images/IMG_20220209_072428.jpg?raw=true" height="400px" width="600px"/>
<br />
[Figure 1: A sample workstation image]
  
* Labelled the images with `labelImg` library.
<br />
<img src="https://github.com/denniesbor/DSI_Twist_Chellenge/blob/2b76763b9c4d6ea852d3fe28953f1a41e7d1e289/images/label_0.png?raw=true" height=height="540px" width="480px">
<br />
[Figure 2: Labelled hazard]
<br />
* Transform the images from xml annotation format to voc
* Transfer learning using tf object detection api. (SSD, FRCNN Resnet 50) - ideas and code from module 1.
* Populate the data with augmentation techniques - rotation, flip, paddinf, cropping, add patches, random (brightness, saturation, hue), graying, resizing.

---
<em />RESULTS:</em >
* No meaningful results after re-engineering data aug params
<br />
<img src="https://github.com/denniesbor/DSI_Twist_Chellenge/blob/assets/images/image.jpg?raw=true" height="430px" width="540px"/>
<br />
[Figure 3: Predictions of potential hazards]
<br />
Tensorboard results.
<img src="https://github.com/denniesbor/DSI_Twist_Chellenge/blob/main/images/results_0.png?raw=true" height="400px" width="540px"/>
---

<em />FAILURE CAUSES:</em >
* Limited training datasets.
* Neglected to label the mouse, laptop and keyboard- Since model is trained on these items would be easy to isolate the risks.
* Resources - 
---
<h3><strong>DEPLOYMENT:</strong ></h3>
<em>CONCEPT:</em > 
<br />

* Deploy the trained model to cloud. 
* Capable of performing realtime video/image detection
* Mail notification in case of potential hazards

----
<em>RESULTS:</em >

* Takes in an image - No video capability
* Slow rendering - Not efficient for real time detection
* High fpr

----
<strong><h3> Resources </h3></strong>

* Code sourced from module 1 project [Help Protect...](https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF)
* Cloud environments - [colab](https://colab.research.google.com/drive/1LvqwyfhgqwMXkn--WLyvbzhvxTr4gNSV#scrollTo=1_geu2gjVSGV) and aws
* [NGINX](https://github.com/nginx/nginx) reverse proxy server
* [Flask API](https://github.com/pallets/flask)






