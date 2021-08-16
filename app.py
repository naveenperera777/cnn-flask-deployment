from flask import Flask, render_template, redirect, request, send_from_directory;
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import cv2
import joblib


app  = Flask(__name__ , template_folder='ui');

app.config["IMAGE_UPLOADS"] = "./static"

 
MODEL_ARCHITECTURE = './Model/model.json'   
MODEL_WEIGHTS = './Model/weights.h5'  

COVID_MODEL_ARCHITECTURE = './Model/covidmodel.json'   ### 
COVID_WEIGHTS = './Model/covidweights.h5'



disease_classes = {0:'Atelectasis',1:'Cardiomegaly', 2 :'Consolidation', 3: 'Edema', 4: 'Effusion',  5: 'Emphysema', 6: 'Fibrosis', 7: 'Hernia', 8: 'Infiltration', 9: 'Mass', 10: 'No Finding', 11 : 'Nodule', 12: 'Pleural_Thickening',13: 'Pneumonia', 14: 'Pneumothorax'}



scalerLoaded = joblib.load('./Model/sclaer.mod')
scalerLoaded.clip = False  
print("Scaler Loaded", scalerLoaded.get_params())


@app.route('/',  methods=["GET", "POST"])
def uploadFile():
    return render_template('index.html')

@app.route('/about',  methods=["GET"])
def about():
    return render_template('about.html')

@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        print("prediction")
        xrayImage = request.files['xrayImage']
        if xrayImage:            
            print("inside")
            filename = xrayImage.filename.split('.')
            print("file", xrayImage.filename)
            xrayImage.save(os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename))
            age = request.form['age']
            gender = request.form['gender']
            print("age", age, "gender", gender)
            scaledAge = scalerLoaded.transform([[int(age)]])
            attributes = [[scaledAge[0][0], int(gender)]]
            print("attributes", attributes)
            attributesArr = np.asarray(attributes)
           
            imagePath = os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename)             

            inputImages = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            inputImages = cv2.resize(inputImages, (128,128))
            inputImages = inputImages[np.newaxis,:,:,np.newaxis]


            inputImages = inputImages / 255.0
            print("last", inputImages.shape)        

            json_file = open(MODEL_ARCHITECTURE)
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

            preds = loaded_model.predict([ attributesArr, inputImages])

            a=np.argmax(preds[0])
            print("predictions", "selected", a , "value ==========> ", disease_classes[a]  , "preds", preds, "pred", preds[0])
            predictionProbability = round(preds[0][a] *100 , 2)
    return render_template(
        'prediction.html', 
        prediction=disease_classes[a],
        fileName=xrayImage.filename,
        predictionProbability=predictionProbability,
        probabilities=preds[0]
        );

@app.route('/covid',  methods=["GET"])
def covid():
    return render_template('covidIndex.html')

@app.route('/covid-prediction', methods=["POST", "GET"])
def covidPrediction():
    if request.method == "POST":
        print("prediction")
        xrayImage = request.files['xrayImage']
        if xrayImage:            
            print("inside")
            filename = xrayImage.filename.split('.')
            print("file", xrayImage.filename)
            xrayImage.save(os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename))
           
            imagePath = os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename)             

            inputImages = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            inputImages = cv2.resize(inputImages, (224,224))
            inputImages = inputImages[np.newaxis,:,:,np.newaxis]


            inputImages = inputImages / 255.0
            print("last", inputImages.shape)        

            json_file = open(COVID_MODEL_ARCHITECTURE)
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

            preds = loaded_model.predict(inputImages)

            a=np.argmax(preds[0])
            disease_classes = {0: 'COVID', 1: 'Normal', 2: 'Viral_Pneumonia'}
            print("predictions", "selected", a , "value ==========> ", disease_classes[a])
            predictionProbability = round(preds[0][a] *100 , 2)
    return render_template(
        'prediction.html', 
        prediction=disease_classes[a],
        fileName=xrayImage.filename,
        predictionProbability=predictionProbability,
        probabilities=preds[0]
    );


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080);

