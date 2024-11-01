# -------------------------------- Importanción de librerías ----------------------------------#
import pandas as pd
import numpy as np
import seaborn as sns
import a_funciones as fn
import cv2
import joblib
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_tuner import RandomSearch

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')
x_val = joblib.load('salidas\\x_val.pkl')
y_val = joblib.load('salidas\\y_val.pkl')

x_train = x_train.astype('float32') ## para poder escalarlo
x_test = x_test.astype('float32') ## para poder escalarlo
x_val = x_val.astype('float32') ## para poder escalarlo

x_train.max()
x_train.min()

x_train /=255
x_test /=255
x_val /=255

x_train.shape
x_test.shape
x_val.shape

# ------------------------------------------ CNN1 ------------------------------------------ #

cnn1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=(2, 2), strides = (2, 2), activation='relu', input_shape= (224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = (2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn1.summary()

cnn1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]
)

# Representación de la arquitectura

keras.utils.plot_model(
    cnn1,
    to_file = 'model.png',
    show_shapes = True,
    show_layer_names = True
)

# Entrenamiento del modelo

history1 = cnn1.fit(
                x_train,
                y_train,
                epochs = 10,
                validation_data = (x_val, y_val)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_auc = cnn1.evaluate(x_val, y_val)

# Gráfica de Recall
plt.plot(history1.history['recall'], label='Recall en el entrenamiento')
plt.plot(history1.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Gráfica de AUC
plt.plot(history1.history['auc'], label='AUC en el entrenamiento')
plt.plot(history1.history['val_auc'], label='AUC en la validación')
plt.title('AUC durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Gráfica de Pérdida
plt.plot(history1.history['loss'], label='Pérdida en el entrenamiento')
plt.plot(history1.history['val_loss'], label='Pérdida en la validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Obtener las predicciones en el conjunto de prueba
y_pred_probs = cnn1.predict(x_val)
y_pred = np.round(y_pred_probs).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm1 = confusion_matrix(y_val, y_pred)
# Visualizar la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# Imprimir el Recall y AUC de la última época para el conjunto de entrenamiento y validación
final_train_recall = history1.history['recall'][-1]
final_val_recall = history1.history['val_recall'][-1]
final_train_auc = history1.history['auc'][-1]
final_val_auc = history1.history['val_auc'][-1]
print(f"Recall en el conjunto de entrenamiento (última época): {final_train_recall:.4f}")
print(f"Recall en el conjunto de validación (última época): {final_val_recall:.4f}")
print(f"AUC en el conjunto de entrenamiento (última época): {final_train_auc:.4f}")
print(f"AUC en el conjunto de validación (última época): {final_val_auc:.4f}")

# Imprimir el reporte de clasificación
report = classification_report(y_val, y_pred, target_names=['Negativo', 'Positivo'])
print("Reporte de Clasificación:")
print(report)

# ------------------------------------------ CNN2 ------------------------------------------ #

cnn2 = tf.keras.Sequential([

    tf.keras.layers.Conv2D(8, kernel_size=(2, 2), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(l2 = 0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]
)

history2 = cnn2.fit(
                x_train,
                y_train,
                batch_size=32,
                epochs = 10,
                validation_data = (x_val, y_val)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_auc = cnn2.evaluate(x_val, y_val)

cnn2.summary()

# Gráfica de Recall
plt.plot(history2.history['recall'], label='Recall en el entrenamiento')
plt.plot(history2.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()


# Gráfica de AUC
plt.plot(history2.history['auc'], label='AUC en el entrenamiento')
plt.plot(history2.history['val_auc'], label='AUC en la validación')
plt.title('AUC durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Gráfica de Pérdida
plt.plot(history2.history['loss'], label='Pérdida en el entrenamiento')
plt.plot(history2.history['val_loss'], label='Pérdida en la validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Obtener las predicciones en el conjunto de prueba
y_pred_probs2 = cnn2.predict(x_val)
y_pred2 = np.round(y_pred_probs2).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm2 = confusion_matrix(y_val, y_pred2)
# Visualizar la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# Imprimir el Recall y AUC de la última época para el conjunto de entrenamiento y validación
final_train_recall2 = history2.history['recall'][-1]
final_val_recall2 = history2.history['val_recall'][-1]
final_train_auc2 = history2.history['auc'][-1]
final_val_auc2 = history2.history['val_auc'][-1]
print(f"Recall en el conjunto de entrenamiento (última época): {final_train_recall2:.4f}")
print(f"Recall en el conjunto de validación (última época): {final_val_recall2:.4f}")
print(f"AUC en el conjunto de entrenamiento (última época): {final_train_auc2:.4f}")
print(f"AUC en el conjunto de validación (última época): {final_val_auc2:.4f}")

# Imprimir el reporte de clasificación
report2 = classification_report(y_val, y_pred2, target_names=['Negativo', 'Positivo'])
print("Reporte de Clasificación:")
print(report)

# -------------------------------- Dropout y regularización -------------------------------- #

#######probar una red con regulzarización L2
reg_strength = 0.001

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.1  

# ------------------------------------------ CNN4 ------------------------------------------ #

def create_cnn_model(input_shape=(100, 100, 3)):
    model = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Cuarta capa convolucional
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Capa de aplanamiento
        Flatten(),
        
        # Capa densa totalmente conectada
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        # Capa de salida
        Dense(1, activation='sigmoid')  # Para clasificación binaria
    ])
    
    # Compilar el modelo
    model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')])

    return model

# Crear el modelo y mostrar su resumen
model = create_cnn_model()
model.summary()

history4 = model.fit(
                x_train,
                y_train,
                batch_size=100,
                epochs = 10,
                validation_data = (x_val, y_val)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_precision, test_auc = model.evaluate(x_val, y_val)

# Gráfica de Recall
plt.plot(history4.history['recall'], label='Recall en el entrenamiento')
plt.plot(history4.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()


# Gráfica de AUC
plt.plot(history4.history['auc'], label='AUC en el entrenamiento')
plt.plot(history4.history['val_auc'], label='AUC en la validación')
plt.title('AUC durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Gráfica de Pérdida
plt.plot(history4.history['loss'], label='Pérdida en el entrenamiento')
plt.plot(history4.history['val_loss'], label='Pérdida en la validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


# ------------------------------------------ CNN5 ------------------------------------------ #
hp = kt.HyperParameters()

# Definición del modelo con Hiperparámetros
def build_model(hp):
    # Definición de hiperparámetros a afinar
    dropout_rate = hp.Float('DO', min_value=0.05, max_value=0.2, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0005, step=0.0001)

    # Creación del modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = (2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = (2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Fijar optimizador en Adam con tasa de aprendizaje definida
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
   
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[metrics.Recall(name='recall'), metrics.AUC(name='auc')]
    )
    
    return model

# Configuración del sintonizador
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective("val_recall", direction="max"),
    max_trials=5,  # Ajusta el número de pruebas para más exploración si es necesario
    overwrite=True,
    directory="my_dir",
    project_name="modelo_afinado"
)

# Ejecución de la búsqueda de hiperparámetros
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=100)

# Selección del mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# Resumen de resultados del tuner
tuner.results_summary()
best_model.summary()

# Evaluación del mejor modelo en el conjunto de prueba
test_loss, test_recall, test_auc = best_model.evaluate(x_val, y_val)
print(f"Recall en prueba: {test_recall}, Precision en prueba: {test_precision}, AUC en prueba: {test_auc}")

# Generación de predicciones
pred_test = (best_model.predict(x_val) >= 0.50).astype('int')

# Guardar el modelo ajustado
#best_model.save('salidas/best_model.h5')


# Entrenamiento del modelo y almacenamiento de las métricas en 'history'
history = best_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=100)

# Imprimir métricas por época
for epoch in range(len(history.history['loss'])):
    print(f"Época {epoch+1}:")
    print(f"  Pérdida en entrenamiento: {history.history['loss'][epoch]}")
    print(f"  Recall en entrenamiento: {history.history['recall'][epoch]}")
    print(f"  AUC en entrenamiento: {history.history['auc'][epoch]}")
    print(f"  Pérdida en validación: {history.history['val_loss'][epoch]}")
    print(f"  Recall en validación: {history.history['val_recall'][epoch]}")
    print(f"  AUC en validación: {history.history['val_auc'][epoch]}")
    print("")

# Graficar las métricas durante el entrenamiento
def plot_perdida(history):
    # Pérdida
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida en validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

def plot_recall(history):
    # Recall
    plt.subplot(2, 2, 2)
    plt.plot(history.history['recall'], label='Recall en entrenamiento')
    plt.plot(history.history['val_recall'], label='Recall en validación')
    plt.title('Recall durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Recall')
    plt.legend()

def plot_auc(history):
    # AUC
    plt.subplot(2, 2, 4)
    plt.plot(history.history['auc'], label='AUC en entrenamiento')
    plt.plot(history.history['val_auc'], label='AUC en validación')
    plt.title('AUC durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Llamada a la función para graficar
plot_perdida(history)
plot_recall(history)
plot_auc(history)

# ------------------------------------------ CNN7 ------------------------------------------ #

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Definir la arquitectura de la red CNN
cnn7 = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Salida binaria para detección de tumor benigno o maligno
])

# Compilar el modelo, aplicando una función de pérdida que permita ajustar la ponderación de las clases
cnn7.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=BinaryCrossentropy(),
    metrics=['recall', 'auc']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True
)

# Definir pesos de clases, dando mayor peso a la clase 1 (maligna)
class_weight = {0: 1.0, 1: 3.0}  # Ajusta según el rendimiento de entrenamiento

history7 = cnn7.fit(
    x_train, y_train, # Datos de entrenamiento
    validation_data=(x_val, y_val),  # Datos de validación
    epochs=20,  # Número de épocas de entrenamiento
    class_weight=class_weight,  # Aplicar el class_weight
    batch_size=32
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_auc = cnn7.evaluate(x_val, y_val)

# Gráfica de Recall
plt.plot(history7.history['recall'], label='Recall en el entrenamiento')
plt.plot(history7.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Gráfica de AUC
plt.plot(history7.history['auc'], label='AUC en el entrenamiento')
plt.plot(history7.history['val_auc'], label='AUC en la validación')
plt.title('AUC durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Gráfica de Pérdida
plt.plot(history7.history['loss'], label='Pérdida en el entrenamiento')
plt.plot(history7 .history['val_loss'], label='Pérdida en la validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Obtener las predicciones en el conjunto de prueba
y_pred_probs7 = cnn7.predict(x_val)
y_pred7 = np.round(y_pred_probs7).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm7 = confusion_matrix(y_val, y_pred7)
# Visualizar la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(cm7, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# Imprimir el Recall y AUC de la última época para el conjunto de entrenamiento y validación
final_train_recall7 = history7.history['recall'][-1]
final_val_recall7 = history7.history['val_recall'][-1]
final_train_auc7 = history7.history['auc'][-1]
final_val_auc7 = history7.history['val_auc'][-1]
print(f"Recall en el conjunto de entrenamiento (última época): {final_train_recall7:.4f}")
print(f"Recall en el conjunto de validación (última época): {final_val_recall7:.4f}")
print(f"AUC en el conjunto de entrenamiento (última época): {final_train_auc7:.4f}")
print(f"AUC en el conjunto de validación (última época): {final_val_auc7:.4f}")

# ------------------------------------------ CNN7 ------------------------------------------ #
# ------------------------------ Afinamiento de la red CNN7 -------------------------------- #

hp = kt.HyperParameters()

def build_model(hp):
    
    dropout_rate = hp.Float('DO', min_value=0.05, max_value= 0.2, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0005, step=0.0001)
    optimizer = hp.Choice('optimizer', ['adam', 'sgd']) ### en el contexto no se debería afinar
   
    ####hp.Int
    ####hp.Choice
    
    cnn7 = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Salida binaria para detección de tumor benigno o maligno
    ])
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
   
    cnn7.compile(
        optimizer=opt, loss="binary_crossentropy", metrics=["Recall", "AUC"],
    ) 
    return cnn7

tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=True, 
    objective=kt.Objective("Recall", direction="max"),
    max_trials=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld", 
)

tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]

tuner.results_summary()
fc_best_model.summary()

# Evaluar el mejor modelo en el conjunto de validación y obtener las métricas
val_loss, val_recall, val_auc = fc_best_model.evaluate(x_val, y_val)
print(f"Evaluación en el conjunto de validación: ")
print(f"Loss: {val_loss:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")

# val_loss, val_auc = fc_best_model.evaluate(x_val, y_val)
pred_val = (fc_best_model.predict(x_test)>=0.50).astype('int')

#################### exportar modelo afinado ##############
fc_best_model.save('salidas\\best_model.keras')

