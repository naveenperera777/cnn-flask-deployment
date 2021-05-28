from flask import Flask, render_template, redirect, request;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import cv2

app  = Flask(__name__);

app.config["IMAGE_UPLOADS"] = "E:\Research\Deployment\Old\static"

 
MODEL_ARCHITECTURE = './Model/model.json'   ###
MODEL_WEIGHTS = './Model/weights.h5'  ###

#Load Model
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

disease_classes = {0:'Atelectasis',1:'Cardiomegaly', 2 :'Consolidation', 3: 'Edema', 4: 'Effusion',  5: 'Emphysema', 6: 'Fibrosis', 7: 'Hernia', 8: 'Infiltration', 9: 'Mass', 10: 'No Finding', 11 : 'Nodule', 12: 'Pleural_Thickening',13: 'Pneumonia', 14: 'Pneumothorax'}

# y_predicted = loaded_model.predict(X_Test_Flatten)
# prediction = np.argmax(y_predicted[0])

print("Model Loaded")

# Get weights into the model
loaded_model.load_weights(MODEL_WEIGHTS)
print("Weights Loaded")

@app.route('/',  methods=["GET"])
def uploadFile():
    return render_template('index.html')

@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        print("prediction")
        xrayImage = request.files['xrayImage']
        if xrayImage:            
            # xrayImage.save('predict.jpg')
            print("inside")

            print("file", xrayImage.filename)
            xrayImage.save(os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename))
            cus = [[0.806452, 0]]
            cusAttr = np.asarray(cus)
            # source_dir = 'E:\\Research\\Deployment\\Old\\static\\186466753_394710685590769_7094925174955611211_n.png' 

            inputImages = []
            imagePath = os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename) 
            print("imagePath", imagePath)
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (128, 128))
            inputImages.append(image)
            inputImages =  np.array(inputImages)
            inputImages = inputImages / 255.0


            preds = loaded_model.predict([ cusAttr, inputImages])

            a=np.argmax(preds[0])
            print("predictions", "selected", a , "value ==========> ", disease_classes[a] , "/n" , "preds", preds)
    return render_template('prediction.html', prediction=disease_classes[a], fileName=xrayImage.filename);


if __name__ == "__main__":
    app.run(debug=True);
