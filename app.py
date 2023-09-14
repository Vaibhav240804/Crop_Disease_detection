import io
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests


app = Flask(__name__)

model = tf.keras.models.load_model('./model/Apple-4-better.h5')

class_list = ["Apple Black rot", "Apple Healthy", "Apple Scab", "Cedar apple rust"]

@app.route('/')
def hello():
    return "Hii there!"

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.form.get('uri')
    print(file)
    if file:
        image = preprocess_image(file)
        
        predictions = model.predict(np.expand_dims(image, axis=0))
        print(predictions)
        # Convert predictions to a format suitable for JSON
        predictions_data = {}
        for i in range(len(class_list)):
            predictions_data[class_list[i]] = float(predictions[0][i])
        
        return jsonify(predictions_data)

    return "No file received"

def preprocess_image(file_url):
    try:
        # Download the image from the URL
        response = requests.get(file_url)
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))

            # Resize the image to (224, 224) and normalize pixel values as needed
            image = image.resize((224, 224))
            # Perform any other preprocessing steps here

            return image
        else:
            print("Failed to download image from URL")
            return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
