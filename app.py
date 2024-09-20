from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Importa CORS
from New_Cnsult import predict_duration  # Importa tu función de predicción

app = Flask(__name__)

# Configurar CORS para permitir todos los orígenes (o un origen específico si es necesario)
CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/')
def index():
    return render_template('index.html')  # Renderiza la página de inicio

@app.route('/form')
def form():
    return render_template('form.html')  # Renderiza el formulario de proyectoss

@app.route('/fprueba')
def fprueba():
    return render_template('fprueba.html')  # Renderiza la página de prueba

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Verificar si los datos vienen en formato form-data o JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extraer los campos del formulario o JSON
        type = int(data.get('type'))
        square_meters = float(data.get('square_meters'))
        weather = int(data.get('weather'))
        soil = int(data.get('soil_state'))
        material = int(data.get('material'))
        design = int(data.get('design'))
        role = int(data.get('role'))
        experience = int(data.get('experience'))
        age = int(data.get('age'))

        # Crear la lista de entrada para el modelo
        input_data = [type, square_meters, weather, soil, material, design, role, experience, age]

        print(f"Datos recibidos: {input_data}")  # Depuración

        # Llamar a la función de predicción
        predicted_duration = predict_duration(input_data)

        # Convertir a float para serializar correctamente
        if predicted_duration is not None:
            return jsonify({'prediction': float(predicted_duration)})  # Convertir a float nativo de Python
        else:
            return jsonify({'error': 'No se pudo generar la predicción.'}), 500

    except Exception as e:
        # Registrar el error para depuración y retornar un error 500
        print(f"Error procesando los datos: {str(e)}")
        return jsonify({"error": "Error interno en el servidor"}), 500

if __name__ == '__main__':
    app.run(debug=True)