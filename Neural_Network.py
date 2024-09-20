import tensorflow as tf
import numpy as np
import os

# Definir los datos de entrenamiento
training_data = np.array([
    # Type, Square Meters, Weather, Soil, Material, Design, Role, Experience, Age
    [1, 711, 1, 1, 1, 1, 1, 1, 48],
    [2, 500, 0, 1, 0, 0, 0, 3, 35],
    [4, 1530, 7, 3, 2, 5, 7, 6, 3],
    [3, 920, 9, 1, 4, 6, 1, 1, 6],
    [2, 1300, 2, 5, 1, 7, 4, 10, 5],
    [5, 1500, 5, 6, 3, 2, 2, 4, 7],
    [1, 400, 3, 2, 5, 3, 8, 3, 4],
    [6, 600, 8, 4, 1, 1, 5, 7, 1],
    [2, 850, 6, 3, 2, 4, 6, 2, 8],
    [1, 300, 10, 2, 3, 8, 1, 5, 2],
    [5, 1100, 1, 5, 4, 6, 3, 8, 9],
    [3, 1700, 4, 1, 1, 2, 9, 6, 3],
    [4, 950, 5, 3, 5, 7, 4, 9, 6],
    [2, 1200, 2, 6, 2, 1, 7, 3, 10],
    [5, 800, 9, 4, 3, 6, 8, 1, 5],
    [1, 500, 6, 5, 1, 8, 4, 10, 4],
    [4, 1500, 3, 2, 2, 5, 2, 4, 8],
    [3, 700, 1, 4, 4, 2, 9, 9, 6],
    [6, 1350, 7, 5, 5, 7, 3, 6, 1],
    [2, 900, 5, 3, 1, 4, 1, 3, 7],
    [1, 800, 4, 6, 2, 3, 5, 7, 3],
    [5, 1600, 2, 1, 4, 5, 6, 8, 9],
    [3, 400, 8, 5, 3, 6, 1, 10, 6],
    [2, 1400, 3, 2, 1, 1, 7, 9, 5],
    [4, 600, 9, 3, 5, 6, 2, 2, 10],
    [1, 200, 10, 4, 3, 8, 4, 1, 2],
    [5, 950, 1, 6, 2, 4, 6, 8, 8],
    [3, 1700, 4, 5, 1, 2, 3, 3, 4],
    [2, 1300, 5, 1, 4, 2, 1, 10, 9],
    [4, 400, 6, 3, 3, 5, 2, 4, 8],
    [6, 1600, 2, 2, 2, 1, 3, 1, 8],
    [5, 700, 3, 4, 5, 6, 8, 1, 5],
    [3, 900, 10, 5, 1, 8, 5, 6, 2],
    [2, 1500, 7, 2, 3, 4, 3, 5, 6],
    [1, 1200, 8, 1, 4, 2, 1, 9, 1],
    [4, 800, 5, 3, 2, 5, 2, 3, 7],
    [6, 1300, 9, 6, 1, 1, 6, 4, 3],
    [5, 500, 1, 4, 3, 3, 1, 5, 4],
    [5, 950, 1, 6, 2, 4, 6, 8, 8],
    [3, 1700, 4, 5, 1, 2, 3, 3, 4],
])

# Definir los resultados esperados (duración en meses, por ejemplo)
output_data = np.array([
    [14], [10], [16], [12], [14], [20], [8], [10], [11], [6], 
    [18], [22], [15], [19], [17], [9], [20], [14], [23], [11], 
    [12], [21], [7], [16], [12], [5], [13], [20], [15], [9], 
    [25], [18], [14], [19], [17], [12], [22], [8], [24], [11]
])

# Crear el modelo de la red neuronal
model = tf.keras.Sequential()

# Añadir capas a la red neuronal
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(9,)))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
model.fit(training_data, output_data, epochs=100, batch_size=1)

# Guardar el modelo entrenado
model_path = 'C:/Users/alexi/OneDrive/Escritorio/Unipoli/Cuatrimestre 3/Estudio Extra/Clean_Neural_Networkpy.py/Results'
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save(os.path.join(model_path, 'model.h5'))
print(f'Modelo guardado en el directorio: {model_path}')

# Función para hacer una predicción
def predict_duration(input_data):
    input_tensor = np.array([input_data])
    result = model.predict(input_tensor)
    print('Predicción:', result[0][0])

# Ejemplo de uso
predict_duration([2, 1300, 5, 1, 4, 2, 1, 10, 9])