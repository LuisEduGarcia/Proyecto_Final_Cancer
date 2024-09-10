from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Configuración del directorio de subida dentro de la carpeta estática
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Asegúrate de que la carpeta exista
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Carga del modelo
modelo_path = 'templates/model/resnet50_fine_tuned_model_40.h5'
model = tf.keras.models.load_model(modelo_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Cargar la imagen y redimensionarla a 256x256
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(256, 256))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Añadir una dimensión para el lote
            
            # Preprocesar la imagen para ResNet50
            img_array = preprocess_input(img_array)  # Normaliza la imagen según los requisitos de ResNet50

            # Realizar la predicción
            predictions = model.predict(img_array)
            prediction = predictions[0][0]  # Suponiendo que el modelo devuelve un array con una sola probabilidad

            # Interpretar el resultado
            if prediction == 0:
                result_message = "El modelo no detecta una alta probabilidad de melanoma maligno."
            else:
                result_message = "El modelo ha detectado una alta probabilidad de melanoma maligno (más del 80%)."
            # URL para la imagen
            image_url = url_for('static', filename=f'uploads/{filename}')

            # Renderizar la página de resultados con la imagen y el mensaje
            return render_template('result.html', image_url=image_url, result_message=result_message)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)