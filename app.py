from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

# loading the model
model = load_model("model.h5")

def preprocess_image(img):
    img_resized = cv2.resize(img, (150, 150))
    img_normalized = img_resized / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(img_normalized)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predicted_class = preprocess_image(img_rgb)

        print("predicted_class : ",predicted_class[0])
        class_labels = ['apple fruit',
                        'banana fruit',
                        'cherry fruit',
                        'chickoo fruit',
                        'grapes fruit',
                        'kiwi fruit',
                        'mango fruit',
                        'orange fruit',
                        'strawberry fruit']# Add the class labels in the order they were used during training
        class_label = class_labels[predicted_class[0]]
        print(class_label)
        return jsonify({'class_label': class_label})
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    #app.run(host = '0.0.0.0',port = 8080)
