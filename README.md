# TensorFlow_Sign_Language_Detection

---

Author: (will be updated later)
* Fathullah Auzan Setyo Laksono (A0040337)
* Rizqullah Fadhil Rafi (A0040334)
* Shabrina Zhafarina Putri (C0040335)
* Puti Syifa Khairani (C0040343)
* I Made Nugraha Pinton (M0040331)
* Immanuel Farhan Sipahutar (M1141448)

---

## Dataset Overview
We will use our *Sign Language* dataset, which was created from combining [American Sign Language Letters Dataset v1 from Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters/1) and our own dataset that we created using our team android devices.
You can find the dataset in [here](https://console.cloud.google.com/storage/browser/sign_language_dataset2)

Each image in the dataset contains objects labeled as one of the following classes: 
* Alphabet from **[A - Y]** without **J** and **Z**, we exclude this two characters as it is required motion to perform.

The dataset contains the bounding-boxes specifying where each object locates, together with the object's label.<br>
Here is some example images from the dataset:

<br/>

<img src="https://storage.googleapis.com/sign_language_dataset2/Documentation/Doc2.png">

---

## Preparing Dataset

### Step 1. Collecting Images
We collect images using our own android mobile devices, since we will use our model on android mobile device. You need to make sure that the dataset that you want to use will represent the image where your model will be deployed.

### Step 2. Labeling Images
Since it is an object detection model, we need to create bounding boxes and label our image, which is a lot easier if you use Google Cloud Platform. Here you'll use the same dataset as the AutoML quickstart you can find more detail about how to prepare your dataset [here](https://cloud.google.com/vision/automl/object-detection/docs/edge-quickstart#preparing_a_dataset).<br>
You will need to create bounding boxes around the object and place a label on top of it, like this image below<br>
<img src="https://storage.googleapis.com/sign_language_dataset2/Documentation/Doc3.png"><br>
In our *Sign Language Dataset*, we have 2014 images with a lot of different angles and different background which are splitted into 24 classes.<br>
You can find more detail about how to prepare a good dataset by simply following this tutorial [here](https://cloud.google.com/vision/automl/object-detection/docs/prepare).<br>
As for our own dataset distribution, it can be represented with the following bar chart image.
<img src="https://storage.googleapis.com/sign_language_dataset2/Documentation/download.png">

### Step 3. Splitting Dataset
Split the dataset into *train*, *validation*, and *test*. You can do it manually or you can leave it to Google Cloud Project which will divide your dataset automatically with 80% for train, 10% for validation, and 10% for test.

### Step 4. Export into CSV Format
The last step is to export your dataset into CSV format which will later be used for training your model. You can do it by simply click the export button and specify the bucket where you want to store it.

---

## Training the Sign Language Detector Model
You can start training the Sign Language Detector model by simply running the **TFLite_Sign_Detection.ipynb** on google colab, which is provided in this repository. <br>

There are six steps to training an object detection model:

### Step 1. Choose an Object Detection Model Archiecture
This tutorial uses the EfficientDet-Lite2 model. EfficientDet-Lite[0-4] are a family of mobile/IoT-friendly object detection models derived from the [EfficientDet](https://arxiv.org/abs/1911.09070) architecture. 

Here is the performance of each EfficientDet-Lite models compared to each others.

| Model architecture | Size(MB)* | Latency(ms)** | Average Precision*** |
|--------------------|-----------|---------------|----------------------|
| EfficientDet-Lite0 | 4.4       | 37            | 25.69%               |
| EfficientDet-Lite1 | 5.8       | 49            | 30.55%               |
| EfficientDet-Lite2 | 7.2       | 69            | 33.97%               |
| EfficientDet-Lite3 | 11.4      | 116           | 37.70%               |
| EfficientDet-Lite4 | 19.9      | 260           | 41.96%               |

<i> * Size of the integer quantized models. <br/>
** Latency measured on Pixel 4 using 4 threads on CPU. <br/>
*** Average Precision is the mAP (mean Average Precision) on the COCO 2017 validation dataset.
</i>

The model architecture are as follows:<br>
<img src="https://storage.googleapis.com/sign_language_dataset2/Documentation/Doc4.png"><br>

EfficientDet-Lite2 Object detection model was trained on COCO 2017 dataset, optimized for TFLite, designed for performance on mobile CPU, GPU, and EdgeTPU.

This model takes input as a batch of three-channel images of variable size. The input tensor is a tf.uint8 tensor with shape [None, height, width, 3] with values in [0, 255]. Where *height = 488 pixels* and *width = 488 pixels*

Since it an object detection model, it produce 4 output. The output dictionary contains:

* **detection_boxes**<br>
a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].

* **detection_scores**<br>
a tf.float32 tensor of shape [N] containing detection scores.

* **detection_classes**<br>
a tf.int tensor of shape [N] containing detection class index from the label file.

* **num_detections**<br>
a tf.int tensor with only one value, the number of detections [N].

### Step 2. Load the Dataset
Model Maker will take input data in the CSV format. Use the `ObjectDetectorDataloader.from_csv` method to load the dataset and split them into the training, validation and test images.

* Training images: These images are used to train the object detection model to recognize salad ingredients.
* Validation images: These are images that the model didn't see during the training process. You'll use them to decide when you should stop the training, to avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting).
* Test images: These images are used to evaluate the final model performance.

You can load the CSV file directly from Google Cloud Storage, but you don't need to keep your images on Google Cloud to use Model Maker. You can specify a local CSV file on your computer, and Model Maker will work just fine.

### Step 3. Train the TensorFlow Model with the Training Data
* The EfficientDet-Lite0 model uses `epochs = 50` by default, which means it will go through the training dataset 50 times. You can look at the validation accuracy during training and stop early to avoid overfitting.
* Set `batch_size = 8`
* Set `train_whole_model=True` to fine-tune the whole model instead of just training the head layer to improve accuracy. The trade-off is that it may take longer to train the model.

You can start the training using **object_detector.create()** function, which is provided in **tflite_model_maker** library.

### Step 4. Evaluate the model with the test data
After training the object detection model using the images in the training dataset, use the remaining images in the test dataset to evaluate how the model performs against new data it has never seen before.<br>
As the default batch size is 64, it will take 4 step to go through the all images in the test dataset.

### Step 5.  Export as a TensorFlow Lite model
Export the trained object detection model to the TensorFlow Lite format by specifying which folder you want to export the quantized model to. The default post-training quantization technique is full integer quantization.

### Step 6.  Evaluate the TensorFlow Lite model
Several factors can affect the model accuracy when exporting to TFLite:
* [Quantization](https://www.tensorflow.org/lite/performance/model_optimization) helps shrinking the model size by 4 times at the expense of some accuracy drop. 
* The original TensorFlow model uses per-class [non-max supression (NMS)](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH) for post-processing, while the TFLite model uses global NMS that's much faster but less accurate.
Keras outputs maximum 100 detections while tflite outputs maximum 25 detections.

Therefore you'll have to evaluate the exported TFLite model and compare its accuracy with the original TensorFlow model.

---

## Android Deployment
To try our Android application, you can download the release version APK file or the Android Studio project files in this repository.<br>

### Step 1. UI/UX
Before making the application, we made a mockup for the application's design guideline. The mockup was made on Adobe XD.

### Step 2. Creating Layout
The next step is making the layout of the app. We simply used Android Studio's activity that has navigation drawer interface because we intended the app to be designed according to the Material Design language guidelines. The only difference is the color palette and the icon that is unique to our app.

### Step 3. Coding the App
In this step, firtly we made sure that all the backend works as intended. We made the connection between the server in Google Cloud and the Android app using [retrofit2](https://square.github.io/retrofit/) for the pattern recognition, and then created a [Room](https://developer.android.com/jetpack/androidx/releases/room) database for storing the scans history on the local device. We tested the backend's functionality using logcat, and after all backend part is working flawlessly we go through designing the frontend side such as the layout with RecyclerView etc.

Here are some screenshots of our app:<br>
<img src="https://cdn.discordapp.com/attachments/504314525873471509/852091042739912754/Screenshot_20210609-144353_Batik_Detection_App.png" width="180" height="320">
<img src="https://cdn.discordapp.com/attachments/504314525873471509/852091042975055882/Screenshot_20210609-144358_Batik_Detection_App.png" width="180" height="320">
<img src="https://cdn.discordapp.com/attachments/504314525873471509/852091043317809172/Screenshot_20210609-144434_Batik_Detection_App.png" width="180" height="320">
<img src="https://cdn.discordapp.com/attachments/504314525873471509/852091043611672586/Screenshot_20210609-144442_Batik_Detection_App.png" width="180" height="320">
<img src="https://cdn.discordapp.com/attachments/504314525873471509/852091043817848842/Screenshot_20210609-144453_Batik_Detection_App.png" width="180" height="320">

---

## Cloud Deployment
You can try to download the HandSign application which are provided in this collaboratory in android studio. We simply following [this tutorial](https://codelabs.developers.google.com/tflite-object-detection-android) for our android deployment.<br>
You can use the **model.tflite** that you get from previous training and save it in the assets folder.<br>

We are using FLASK for the web framework since it the language used for FLASK is Python which the cloud developers in the team is already fond of.<br>
To deploy web app API, we use the App Engine service from Google Cloud Platform so that the android.<br>

### Files Uploaded to the App Engine
The files that is uploaded to the App Engine includes : 

### Step 2. Creating Layout
The next is to implement our UI/UX to the real thing, by making the layout of the application in android studio. Because of the slide bar menu in our application, we use the slide bar menu activity template from android studio and modify it as we need. After finishing the three menu that consist of Home, Settings and Information, next is to make the activity for each feature. Since we only got 1 month to finish this project, we mainly focus on Sign language translator feature meanwhile the other feature is display as "Under Development" for future improvement. 

### Step 3. Coding the App
For this step before start to coding the sign language translator feature, we make sure the other feature works first, making Intent for every button and directing it to the correct activity or fragment. Then we start to coding sign language translator feature by preparing the input for the app. This app can take picture with the android camera or take it from the android galery folder. The choosen picture than displayed on the application screen so that can be analyzed by the model then outputing the result.

Since we follow the tutorial from the official documentation, this object detection step mostly similar to that. We make a function to make the bounding box for showing the translator result and anothe function to return the image to be displayed in the application, then a function to call the tflite model with the input is Bitmap of the image that alread be converted before. 

Here is the some images of how our application looks like.<br>
<img src="https://storage.googleapis.com/sign_language_dataset2/Documentation/Doc5.png">

And here is the mAP (Mean Average Precision) on each characters:<br>
<img src="https://storage.googleapis.com/sign_language_dataset2/Documentation/Doc6.png">

---
