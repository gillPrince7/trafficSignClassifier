from flask import Flask, request, render_template
from util import classify_image
import numpy as np
import cv2
from io import BytesIO
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded.", 400
    image = request.files['image']
    filename = str(image.filename)
    img = cv2.imread(filename)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    pred_class = ['Construction zone sign','do not enter','No passing zone sign','One way sign','Pedestrian crossing sign','railway crossing sign','School zone','Speed limit sign','stop photos','yield sign']

    from keras.models import load_model
    model_final = load_model('model.h5')
    predictions = model_final.predict(img)
    class_index = np.argmax(predictions)
    pred_class = pred_class[class_index]
    return (pred_class)

   # result = classify_image(img)
   # print(result)
    # do something with the image, such as classify it
   # return "type(image)"

if __name__ == '__main__':
    app.run(debug=True)
