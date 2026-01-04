from flask import Flask, render_template, request
import os


from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np


model = VGG16()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    IMAGE_FOLDER =  './images/'
    image = request.files['image']
    if image.filename == '':
        return render_template('index.html', result='No file selected')
    from werkzeug.utils import secure_filename
    filename = secure_filename(image.filename)
    imagepath = os.path.join(IMAGE_FOLDER, filename)
    image.save(imagepath)

    img = load_img(imagepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    result1 = [p[1] for p in decoded_predictions]
    result = []
    result.append(imagepath[2:])
    result.append(result1)
    return render_template('index.html', result=result)




if __name__ == '__main__':
    app.run(debug=True, port=5000)
    


