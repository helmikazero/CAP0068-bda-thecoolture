import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
from imageio import imread
import base64
import cv2
from flask import Flask, request, Response, jsonify
import jsonpickle
import urllib.parse
from firebase_admin import credentials, firestore, initialize_app, storage
# import os

app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate("batikrecog-84-firebase-adminsdk-jbbex-c097ae4770.json")
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('debugging2')


model = tf.keras.models.load_model('batik_dropout.h5')

resizevalue = (150,150)

@app.route("/predict/pcpy", methods=['POST'])
def predict4():
    r = request
    print("--------------POST RECEIVED-------------")
    print(r.data[:20])
    print("--------------BASE64 RECEIVED-------------")

    img = imread(io.BytesIO(base64.b64decode(r.data)))
    
    print("image received")
    
    nama_batik = ["Ceplok","Kawung","Lereng","Mix","Nitik","Parang"]
    index = 0

    #jika ada pre-process taro disini
    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    im_resized = im_pil.resize((150,150))
    img_array = image.img_to_array(im_resized) / 255.0
    image_array_expanded =np.expand_dims(img_array,axis=0)

    img_vstack = np.vstack([image_array_expanded])

    #-------------
    hasil_prediksi = model.predict(img_vstack, batch_size=0)
    
    fullReport = {"prediction": []}
    
    for hasil in hasil_prediksi[0]:
        
        curr_pred = {"label": nama_batik[index], "probability": hasil.item()}
        fullReport["prediction"].append(curr_pred)

        index = index + 1
    

    index = 0

    print(fullReport)
    
    response_pickled = jsonpickle.encode(fullReport)
    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route("/predict/full", methods=['POST'])
def predict3():
    r = request

    print("-----------------------------------------")
    # print(r.data)
    # paper0 = open("rdata.txt","x")
    # paper0.write(str(r.data))
    # paper0.close()
    # storage.child("b64andro/b64a.txt").put("rdata.txt")
    # if os.path.exists("rdata.txt"):
    #     os.remove("rdata.txt")
    print("--------------POST RECEIVED-------------")
    print(r.data[:20])
    print(str(r.data[5:20]))
    print(type(r.data[5:20].decode()))
    print("--------------BASE64 RECEIVED-------------")

    cleandata = r.data[5:].decode()
    cleandata = urllib.parse.unquote(cleandata)

    cleandata_debug = cleandata[:20]

    todo_ref.add({'rdata':str(r.data),'cleandata':str(cleandata)})
    print("---------CLEANING RESULT----------")
    print("FORMAT:{} CLEAN={}".format(type(cleandata),cleandata_debug))
    print("---------XXXXXXXXXXXXXXX----------")

    pdata0 = base64.b64decode(str(cleandata))
    pdata1 = io.BytesIO(pdata0)
    img = imread(pdata1)

    # img = imread(io.BytesIO(base64.b64decode(cleandata)))
    
    
    print("image received")
    
    nama_batik = ["Ceplok","Kawung","Lereng","Mix","Nitik","Parang"]
    index = 0

    #jika ada pre-process taro disini
    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    im_pil = Image.fromarray(cvt_image)

    im_resized = im_pil.resize((150,150))
    img_array = image.img_to_array(im_resized) / 255.0
    image_array_expanded =np.expand_dims(img_array,axis=0)
    
    img_vstack = np.vstack([image_array_expanded])

    #-------------
    hasil_prediksi = model.predict(img_vstack, batch_size=0)
    
    fullReport = {"prediction": []}
    
    for hasil in hasil_prediksi[0]:
        
        curr_pred = {"label": nama_batik[index], "probability": hasil.item()}
        fullReport["prediction"].append(curr_pred)

        index = index + 1


    index = 0

    print(fullReport)
    
    response_pickled = jsonpickle.encode(fullReport)
    return Response(response=response_pickled, status=200, mimetype="application/json")



@app.route("/predict_b64", methods=['POST'])
def predict2():
    r = request
    
    # print(r.data)
    # print(type(r))
    # b64_string = base64.b64decode(r.data)

    img = imread(io.BytesIO(base64.b64decode(r.data)))
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    print("image received")
    
    nama_batik = ["Ceplok","Kawung","Lereng","Mix","Nitik","Parang"]
    index = 0
    hi_pred = 0
    hi_index = 0




    #jika ada pre-process taro disini
    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    im_resized = im_pil.resize((150,150))
    img_array = image.img_to_array(im_resized) / 255.0
    image_array_expanded =np.expand_dims(img_array,axis=0)

    img_vstack = np.vstack([image_array_expanded])


    #-------------
    hasil_prediksi = model.predict(img_vstack, batch_size=0)
    
    
    
    for hasil in hasil_prediksi[0]:
        if(hasil*100 > hi_pred):
            hi_pred = hasil*100
            hi_index = index
        print("{} = {}".format(nama_batik[index],hasil*100))
        index = index + 1
    
    response = {'label': nama_batik[hi_index]}
    #response = {'label': 'response is fine'}

    index = 0
    
    response_pickled = jsonpickle.encode(response)
    print(response_pickled)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route("/predict", methods=['POST'])
def predict():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("image received")
    
    nama_batik = ["Ceplok","Kawung","Lereng","Mix","Nitik","Parang"]
    index = 0
    hi_pred = 0
    hi_index = 0


    #jika ada pre-process taro disini
    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    im_resized = im_pil.resize((150,150))
    img_array = image.img_to_array(im_resized) / 255.0
    image_array_expanded =np.expand_dims(img_array,axis=0)

    img_vstack = np.vstack([image_array_expanded])


    #-------------
    hasil_prediksi = model.predict(img_vstack, batch_size=0)
    
    
    
    for hasil in hasil_prediksi[0]:
        if(hasil*100 > hi_pred):
            hi_pred = hasil*100
            hi_index = index
        index = index + 1
    
    response = {'label': nama_batik[hi_index]}
    #response = {'label': 'response is fine'}

    index = 0
    
    response_pickled = jsonpickle.encode(response)
    print(response_pickled)
    return Response(response=response_pickled, status=200, mimetype="application/json")


    
@app.route("/")
def default():
    return "<h1>Welcome to our server !!</h1>"
    





if __name__ == '__main__':
    app.run(threaded=True, port=8080)