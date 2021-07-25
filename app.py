from flask import Flask, render_template, redirect, request;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
# from pickle import dump, load
# from sklearn.preprocessing import MinMaxScaler
import joblib

# from sklearn.externals import joblib 

app  = Flask(__name__);

app.config["IMAGE_UPLOADS"] = "E:\Research\Deployment\CNN Flask Deployment\static"

 
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

# Load Data Scaling
# scalerLoaded = joblib.load(open('E:\Research\Deployment\CNN Flask Deployment\Model\scaler.pkl', 'rb'))
scalerLoaded = joblib.load('./Model/scaler.mod')
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
            # xrayImage.save('predict.jpg')
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
            # source_dir = 'E:\\Research\\Deployment\\Old\\static\\186466753_394710685590769_7094925174955611211_n.png' 

            inputImages = []
            imagePath = os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename) 
            print("imagePath", imagePath)
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("graybefore", gray.shape, image.shape)
            image = cv2.resize(gray, (128, 128))
            inputImages.append(image)            
            print("imageshape", image.shape, "gray", gray.shape)
            inputImages =  np.array(inputImages)
            # img = image.load_img(imagePath, target_size=(128, 128),color_mode='grayscale')
            # img = image.img_to_array(img)
            # inputImages.append(img)
            # inputImages = np.array(inputImages)
            inputImages = inputImages / 255.0
            # To delete file
            # os.remove(os.path.join(app.config["IMAGE_UPLOADS"], xrayImage.filename))

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


if __name__ == "__main__":
    app.run(debug=True);

