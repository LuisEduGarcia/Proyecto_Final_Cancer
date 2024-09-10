from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Cargar el Modelo de Metadatos y el Dataset
metadata_model_path = 'static/model/metadata_model.pkl'
metadata_model = joblib.load(metadata_model_path)

image_model_path = 'static/model/prueba_modelo_encadenado.pkl'
image_model = load_model(image_model_path)

df = pd.read_csv('static/data/dataset_flask.csv')

def get_metadata(df, image_id):
    image_metadata = df[df['isic_id'] == image_id]
    if image_metadata.empty:
        return None
    return image_metadata

def predict_metadata(df, metadata_model, image_id):
    image_metadata = get_metadata(df, image_id)
    if image_metadata is None:
        return None
    X_metadata = image_metadata.drop(columns=['isic_id', 'target'])
    metadata_prediction = metadata_model.predict_proba(X_metadata)[:, 1]  # Probabilidad de la clase 1
    return metadata_prediction

def preprocess_image(image_path, size=(256, 256)):
    img = load_img(image_path, target_size=size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Normalización específica de ResNet50
    img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión para el batch
    return img_array

def predict_image_with_metadata(image_path, image_model, metadata_pred):
    image_array = preprocess_image(image_path)
    combined_input = [image_array, np.array(metadata_pred).reshape(-1, 1)]
    final_prediction = image_model.predict(combined_input)
    return final_prediction

def display_patient_info(df, isic_id):
    # Obtener la información del paciente
    patient_info = df[df['isic_id'] == isic_id][['age_approx', 'count_per_patient', 'clin_size_long_diam_mm', 'volume_approximation_3d', 'color_asymmetry_index']].to_dict(orient='records')
    if patient_info:
        # Redondear el índice de asimetría del color a 3 decimales
        patient_info[0]['color_asymmetry_index'] = round(patient_info[0]['color_asymmetry_index'], 3)
    return patient_info

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            
            # Realizar la predicción
            image_id = os.path.splitext(file.filename)[0]
            metadata_pred = predict_metadata(df, metadata_model, image_id)
            if metadata_pred is not None:
                final_prediction_raw = predict_image_with_metadata(file_path, image_model, metadata_pred)
                final_prediction_raw = float(final_prediction_raw)  # Convertir a float para usar en el template
                
                # Obtener información del paciente
                patient_info = display_patient_info(df, image_id)

                # Calcular y redondear la confianza
                confidence = final_prediction_raw if final_prediction_raw >= 0.5 else 1 - final_prediction_raw
                image_confidence = round(confidence * 100, 3)  # Redondear a 3 decimales
                final_prediction_percentage = round(final_prediction_raw, 3)  # Redondear a 3 decimales

                # Crear el mensaje de resultado
                if final_prediction_raw < 0.5:
                    result_message = f"El modelo no detecta una alta probabilidad de melanoma maligno." #result_message = f"El modelo no detecta una alta probabilidad de melanoma maligno. Con un resultado de: {final_prediction_percentage}"
                else:
                    result_message = f"El modelo ha detectado una alta probabilidad de melanoma maligno."#result_message = f"El modelo ha detectado una alta probabilidad de melanoma maligno. Con un resultado de: {final_prediction_percentage}"

                return render_template('result.html',
                                       image_url=url_for('static', filename='uploads/' + file.filename),
                                       filename=file.filename,
                                       result_message=result_message,
                                       image_confidence=image_confidence,
                                       patient_info=patient_info)
            else:
                return "No se encontró metadatos para la imagen.", 400
    return render_template('upload.html')
if __name__ == '__main__':
    app.run(debug=True)