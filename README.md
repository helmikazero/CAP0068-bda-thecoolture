# thecoolture_bda
Repository for Batik Detection App by The Coolture Team, made for Bangkit 2021 Capstone Project.

Contributors for this application's development:
- Mobile Development: [Albari Berki Pradhana](https://github.com/albarkip) , [Fachri Veryawan Mahkota](https://github.com/mahkota)
- Cloud: [Helmika Mahendra Priyanto](https://github.com/helmikazero), [Meidy Dwi Larasati](https://github.com/meidydwilarasati)
- Machine Learning: [Natasya Ulfha](https://github.com/tasyaulfha), [Yusuf Nur Wahid](https://github.com/ynw99)

For the Machine Learning side of this project, we're inspired by this Batik pattern recognition [repository](https://github.com/yohanesgultom/deep-learning-batik-classification). We also use transfer learning method to get our model better. You can refer to this [medium](https://towardsdatascience.com/transfer-learning-for-image-classification-using-tensorflow-71c359b56673) to learn about it.

---

## Machine Learning

ML here

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
We are using FLASK for the web framework since it the language used for FLASK is Python which the cloud developers in the team is already fond of.<br>
To deploy web app API, we use the App Engine service from Google Cloud Platform so that the android.<br>

Which then we received the main url for the deployed API in the App Engine : <br>
https://batikrecog-b21-cap0068.et.r.appspot.com/ <br>

<img src="https://cdn.discordapp.com/attachments/834812570434535494/852108725040447518/unknown.png"> > <br>

### Files Uploaded to the App Engine
The files that is uploaded to the App Engine includes : <br>
* app.yml - a file that is required by App Engine to be included that consist of the runtime that is used which here we are using Python 3.8
* batik_dropout.h5 - the model that is given by the Machine Learning devs
* batikrecog-84-firebase-adminsdk-jbbex-c097ae4770.json - the authorization key required by the Firebase to access the Firebase
* main.py - the main python file that consists all the API scripts that includes the REST API routes, keras model load, and openCV image processing.
* requirements.txt - also a file required by App Engine that lists all the python library dependencies for the API

### Steps On Deploying the API to App Engine
1. Uploading all of the files to Cloud Shell in GCP
2. Create App in App Engine
3. Deploy the files in the Cloud Shell to GCP App Engine with `gcloud app deploy --no-cache` command with the `--no-cache` part for ensuring of not using the previous cached library if exist.

### Routes Available in the API
There are severals routes that is provided in the API for different purposes : <br>
* `/predict/pcpy` - Meant for receiving POST request from python file that behaves similar to request from Android that sends image and expecting prediciton response
* `/predict/full` - The main route meant for the Android Batik Recognition APP which includes the process of cleaning the received data from android that is quoted.
* `/predict_b64` - Testing routes for debugging the base64 decode-encode process
* `/` - route that expects no request and returns "Welcome to the Server" message for checking the deployment straight from the base URL in case the app is not yet deployed.

### Problems Encountered in Development Process

#### 1. Massive Size of the Whole Deployment
With all the libraries and the model. The first version of the deployment can exceeds the size of 2150 MB which exceeds the maximum size of the "F4_1G" which is already the instance class that has the highest deploy size. In order to solve the problem, we change some of the libraries that have "lighter version" such is the tensorflow that have "tensorflow-cpu" version and opencv-python with "headless-opencv-python" version. <br>

#### 2. Unexpected Quoted Format of the Data Sent by the Android APP
First version of the API expects in request that in a form of base64 encoded bytes that looks like this (only first 20 letters of the whole encoded data) :
`b'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB` <br>

But the received data from the Android looks like this :
`data=%2F9j%2F4AAQSkZJRgABAQAAAQABAAD%2F2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB%0AAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH%2` <br>

So, in order to solve this problem. We included quote cleaning function provided by urllib library - `urrlib.unquote()` then remove the first 5 characters which is the `data=`.<br>

And so after those post-processing we can input the clean base64 bytes to the decoder.<br>

#### 3. Methods of Sending the Image to the API in the Cloud
Sending the whole raw file of image that is in the form of jpeg or png is not possible through the REST API (as far as we know). <br>

There are several methods that is available in the table for the team : <br>
1. Breaking down the image into cv2 image format from the Android and sending it to the cloud.
2. Encoding the image into base64 bytes format and sending it to the cloud. <br>

We chose NOT to use the first option because the OpenCV module will increase the Android APP size for the Batik App. So, we chose the second option of encoding the image into base64 format before sending it to the cloud. <br>

#### 4. GCP App Engine That Sometimes turned OFF
As far as we know, the App Engine from GCP has its own mind of turning the deployment OFF if there is no activity for several moments. It causes sudden request from the Android App to trigger the timeout limit due to waiting the App Engine to start everytime it is turned OFF. <br>

---
