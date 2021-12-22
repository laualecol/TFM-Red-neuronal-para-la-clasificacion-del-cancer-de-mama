from flask import Flask
from flask import render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import numpy as np
from flask import jsonify
import logging
from flask import send_from_directory
import cv2
import pydicom as pdicom


UPLOAD_FOLDER = "/Users/laura/anaconda3/envs/tfm"
DIR = "/Users/laura"
ALLOWED_EXT = set(['dcm'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route("/")
def index():
    return render_template('index.html')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
modelo = model_from_json(loaded_model_json)
#model.load_weights('model.h5py')


labels={0:'Benigno', 1:'Maligno'}

@app.route("/", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if request.files.get('file'):
            img_requested = request.files['file'].read()
            img = pdicom.dcmread(io.BytesIO(img_requested))
            img = img.pixel_array
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = np.array(img)
            img = cv2.resize(img, (256, 256))
            img = img/255
            img = np.expand_dims(img,axis=0)
            prediction = modelo.predict(img) 
            result = np.argmax(prediction,axis=1)[0]
            accuracy = float(np.max(prediction,axis=1)[0])
            label= labels[result]    
        
            print(prediction,result,accuracy)
            response = {'prediction': {'result': label,'accuracy': accuracy}}
            return render_template('index.html', 
                prediction_text = "Con una precisión del {0} la imagen que ha seleccionado es un tumor de mama {1}".format(response["prediction"]["accuracy"],
                response["prediction"]["result"]))



if __name__ == "__main__":
    app.run(debug = True,use_reloader=False, port = '2301')