import tensorflow as tf
import numpy as np

# Ruta del modelo guardado
model_path = 'C:/Users/alexi/OneDrive/Escritorio/Unipoli/Cuatrimestre 3/Estudio Extra/Prototipo_Final/Results/model.h5'

# Cargar el modelo guardado en formato .h5
model = tf.keras.models.load_model(model_path)

def predict_duration(input_data):
    """
    Función que hace una predicción usando el modelo cargado.
    :param input_data: lista de características de entrada (debe ser de tamaño [9])
    :return: predicción de la duración del proyecto
    """
    try:
        input_data = np.array(input_data).reshape(1, -1)  # Asegurarse de que el input tenga forma (1, 9)
        prediction = model.predict(input_data)
        return prediction[0][0]  # Retorna la predicción como un valor escalar
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return None

if __name__ == "__main__":
    # Ejemplo de entrada (puedes cambiar los valores según el input que quieras predecir)
    input_example = [2, 1400, 5, 1, 4, 2, 1, 10, 2]  # Cambia estos valores para hacer predicciones con otros inputs
    result = predict_duration(input_example)
    print(f'Predicción de duración del proyecto: {result:.2f} meses')