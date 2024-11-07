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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras_tuner import RandomSearch

#Generar los archivos
x_train, y_train, x_test, y_test, x_val, y_val = fn.imag_array()

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



fn.plot_recall(history1)
fn.plot_auc(history1)
fn.plot_perdida(history1)

# Obtener las predicciones en el conjunto de prueba
y_pred_probs = cnn1.predict(x_val)
y_pred = np.round(y_pred_probs).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm1 = confusion_matrix(y_val, y_pred)
fn.matriz(cm1)

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

fn.plot_recall(history2)
fn.plot_auc(history2)
fn.plot_perdida(history2)

# Obtener las predicciones en el conjunto de prueba
y_pred_probs2 = cnn2.predict(x_val)
y_pred2 = np.round(y_pred_probs2).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm2 = confusion_matrix(y_val, y_pred2)
# Visualizar la matriz de confusión
fn.matriz(cm2)

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

# ------------------------------------------ CNN7 ------------------------------------------ #

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
class_weight = {0: 1.0, 1: 2.0}  

history7 = cnn7.fit(
    x_train, y_train, 
    validation_data=(x_val, y_val),  
    epochs=10,  
    class_weight=class_weight,  
    batch_size=32
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_auc = cnn7.evaluate(x_val, y_val)

fn.plot_recall(history7)
fn.plot_auc(history7)
fn.plot_perdida(history7)

# Obtener las predicciones en el conjunto de prueba
y_pred_probs7 = cnn7.predict(x_val)
y_pred7 = np.round(y_pred_probs7).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm7 = confusion_matrix(y_val, y_pred7)

# Visualizar la matriz de confusión
fn.matriz(cm7)

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


class_weight = {0: 1.0, 1: 2.0}

# Configuración de hiperparámetros
hp = kt.HyperParameters()

# Definición del modelo
def build_model(hp):
    dropout_rate = hp.Float('DO', min_value=0.05, max_value=0.2, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0005, step=0.0001)
    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])

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
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    cnn7.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
    )
    return cnn7

# Configuración del tuner
tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=True, 
    objective=kt.Objective("val_recall", direction="max"),
    max_trials=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld", 
)

# Búsqueda de hiperparámetros
tuner.search(
    x_train, y_train,
    epochs=20,
    validation_data=(x_val, y_val),
    batch_size=100,
    class_weight=class_weight
)

# Obtener el mejor modelo y evaluación
fc_best_model = tuner.get_best_models(num_models=1)[0]

# Resumen de los resultados y del mejor modelo
tuner.results_summary()
fc_best_model.summary()

# Evaluar el mejor modelo en el conjunto de validación
val_loss, val_recall, val_auc = fc_best_model.evaluate(x_val, y_val, verbose=0)
print(f"Evaluación en el conjunto de validación:")
print(f"Loss: {val_loss:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")

# Realizar predicciones sobre el conjunto de validación
pred_val = (fc_best_model.predict(x_val) >= 0.50).astype('int')


#################### exportar modelo afinado ##############
fc_best_model.save('salidas\\best_model.keras')