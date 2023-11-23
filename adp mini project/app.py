import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Disable eager execution
# tf.compat.v1.disable_eager_execution()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Define a custom object for BatchNormalization layer
custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}

# Load the model with custom_objects
model = tf.keras.models.load_model('adp.h5', custom_objects=custom_objects)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict1():
    return render_template('alzpre.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(180, 180))
        x = image.img_to_array(img)
        X = np.expand_dims(x, axis=0)
        # img_data = preprocess_input(X)
        # with tf.compat.v1.Session() as sess:
        #     with tf.compat.v1.Graph().as_default():
        #         set_session(sess)
        # prediction = model.predict(X)[0][0][0]
        prediction = model.predict(X)[0][0]
        print("Predicted labels:", prediction)  # Move the print statement here
        print("Predicted label index shape:", prediction.shape)


        print(prediction)
        if prediction == 0:
            text = "Mild Demented"
        elif prediction == 1:
            text = "Moderate Demented"
        elif prediction == 2:
            text = "Non Demented" 
        else:
            text = "Very Mild Demented"
        return render_template('alzpre.html', prediction=text)

    return render_template('alzpre.html')
 
if __name__ == "__main__":
    app.run(debug=True)
