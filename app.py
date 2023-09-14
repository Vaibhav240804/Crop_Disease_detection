import io
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests


app = Flask(__name__)


class_list_apple = ["Apple Black rot", "Apple Healthy", "Apple Scab", "Cedar apple rust"]


causes_list_apple = {
    "Apple Black rot":{
        "cause": [
            "Fungle Infection : Apple Black Rot is primarily caused by a fungal pathogen called Botryosphaeria obtusa",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
            "Infected Debris: The fungus can overwinter in infected debris from the previous season.",
            "Insects: Insects can spread the disease from infected trees to healthy trees."
        ],
        "prevention": [
            "Fungicides: Apply appropriate fungicides to protect the apple trees. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the orchard and destroy it. This will help reduce the spread of the disease.",
            "Insecticides: Apply appropriate insecticides to control insects that can spread the disease.",
            "Cultural Practices: Prune trees to improve air circulation and reduce humidity. This will help reduce the spread of the disease."
        ],
    },
    "Apple Healthy":{
        "cause": "Healthy",
        "prevention": "Healthy",
    },
    "Apple Scab":{
        "cause": [
            "Fungal Infection: Apple Scab is caused by the fungus Venturia inaequalis.",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
        ],
        "prevention": [
            "Fungicides: Apply appropriate fungicides to protect the apple trees. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the orchard and destroy it. This will help reduce the spread of the disease.",
        ]
    },
    "Cedar apple rust":{
        "cause": [
            "Fungal Infection: Cedar Apple Rust is caused by the fungus Gymnosporangium juniperi-virginianae.",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
            "Alternate Hosts: The disease requires two hosts, apple trees, and Easter Red Cedar trees, to complete its life cycle.",
        ],
        "prevention": [
            "Fungicides: Apply fungicides as a preventive measure before symptoms appear. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the orchard and destroy it. This will help reduce the spread of the disease.",
            "Remove alternative hosts: Remove Easter Red Cedar trees from the vicinity of the orchard. This will help reduce the spread of the disease.",
        ]
    }
    
}

@app.route('/')
def hello():
    return "Hii there!"

@app.route('/apple', methods=['POST'])
def apple():

    file = request.form.get('uri')
    print(file)
    if file:
        image = preprocess_image(file)
        model = tf.keras.models.load_model('./model/Apple-4-better.h5')
        
        predictions = model.predict(np.expand_dims(image, axis=0))
        print(predictions)

        predictions_data = {}

        for i in range(len(class_list_apple)):
            
            predictions_data[class_list_apple[i]] = float(predictions[0][i])
        predictions_data["causes_Prev"] = causes_list_apple[class_list_apple[np.argmax(predictions)]]
        
        return jsonify(predictions_data)

    return "No file received"


class_list_bellpaper = ["Bell pepper Bacterial spot", "Bell pepper Healthy"]

causes_list_bellpaper = {
    "Bell pepper Bacterial spot":{
        "cause": [
            "Bacteria: Bacterial spot is caused by the bacteria Xanthomonas campestris pv. vesicatoria.",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
            "Infected Debris: The bacteria can overwinter in infected debris from the previous season.",
            "Insects: Insects can spread the disease from infected plants to healthy plants."
        ],
        "prevention": [
            "Fungicides: Apply appropriate fungicides to protect the plants. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the field and destroy it. This will help reduce the spread of the disease.",
            "Insecticides: Apply appropriate insecticides to control insects that can spread the disease.",
            "Cultural Practices: Prune plants to improve air circulation and reduce humidity. This will help reduce the spread of the disease."
        ],
    },
    "Bell pepper Healthy":{
        "cause": "Healthy",
        "prevention": "Healthy",
    }
}

@app.route('/bellpaper', methods=['POST'])
def bellpaper():
    file = request.form.get('uri')
    print(file)
    if file:
        image = preprocess_image(file)
        
        model = tf.keras.models.load_model('./model/Bellpepper-1.h5')
        predictions = model.predict(np.expand_dims(image, axis=0))
        print(predictions)
        # Convert predictions to a format suitable for JSON
        predictions_data = {}
        for i in range(len(class_list_bellpaper)):
            predictions_data[class_list_bellpaper[i]] = float(predictions[0][i])

        # return predictions along with cure and causes of possible disease
        predictions_data["causes_Prev"] = causes_list_bellpaper[class_list_bellpaper[np.argmax(predictions)]]
        
        return jsonify(predictions_data)

    return "No file received"


class_list_citrus = ["Citrus Black spot"," Citrus canker","Citrus greening", "Citrus Healthy"]

causes_list_citrus = {
    "Citrus Black spot":{
        "cause": [
            "Fungle Infection : Citrus Black spot is primarily caused by a fungal pathogen called Guignardia citricarpa",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
            "Infected Debris: The fungus can overwinter in infected debris from the previous season.",
            "Insects: Insects can spread the disease from infected trees to healthy trees."
        ],
        "prevention": [
            "Fungicides: Apply appropriate fungicides to protect the apple trees. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the orchard and destroy it. This will help reduce the spread of the disease.",
            "Insecticides: Apply appropriate insecticides to control insects that can spread the disease.",
            "Cultural Practices: Prune trees to improve air circulation and reduce humidity. This will help reduce the spread of the disease."
        ],
    },
    "Citrus canker":{
        "cause": [
            "Bacteria: Citrus canker is caused by the bacteria Xanthomonas citri subsp. citri.",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
            "Infected Debris: The bacteria can overwinter in infected debris from the previous season.",
            "Insects: Insects can spread the disease from infected plants to healthy plants."
        ],
        "prevention": [
            "Fungicides: Apply appropriate fungicides to protect the plants. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the field and destroy it. This will help reduce the spread of the disease.",
            "Insecticides: Apply appropriate insecticides to control insects that can spread the disease.",
            "Cultural Practices: Prune plants to improve air circulation and reduce humidity. This will help reduce the spread of the disease."
        ],
    },
    "Citrus greening":{
        "cause": [
            "Bacteria: Citrus greening is caused by the bacteria Candidatus Liberibacter asiaticus.",
            "Environmental Conditions: Warm and wet conditions during the growing season create favorable conditions for the disease to spread.",
            "Infected Debris: The bacteria can overwinter in infected debris from the previous season.",
            "Insects: Insects can spread the disease from infected plants to healthy plants."
        ],
        "prevention": [
            "Fungicides: Apply appropriate fungicides to protect the plants. Timing is crucial, so follow recommended schedules.",
            "Pruning: Remove infected branches and fruit to reduce the spread of the disease.",
            "Sanitation: Remove infected debris from the field and destroy it. This will help reduce the spread of the disease.",
            "Insecticides: Apply appropriate insecticides to control insects that can spread the disease.",
            "Cultural Practices: Prune plants to improve air circulation and reduce humidity. This will help reduce the spread of the disease."
        ],
    },
    "Citrus Healthy":{
        "cause": "Healthy",
        "prevention": "Healthy",
    }
}

@app.route('/citrus', methods=['POST'])
def citrus():
    file = request.form.get('uri')
    print(file)
    if file:
        image = preprocess_image(file)
        
        model = tf.keras.models.load_model('./model/Citrus-1.h5')
        predictions = model.predict(np.expand_dims(image, axis=0))
        print(predictions)
        # Convert predictions to a format suitable for JSON
        predictions_data = {}
        for i in range(len(class_list_citrus)):
            predictions_data[class_list_citrus[i]] = float(predictions[0][i])

        # return predictions along with cure and causes of possible disease
        predictions_data["causes_Prev"] = causes_list_citrus[class_list_citrus[np.argmax(predictions)]]
        
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
