
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

cnn1 = keras.models.Sequential()

# Definición de la primera capa convolucional

cnn1.add(
    keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        strides = (2, 2),
        activation = 'relu',
        input_shape = (100, 100, 3)    
    )

)

# Definición de la capa de agrupación

cnn1.add(
    keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides = (2, 2)
    )
)

# Definición de la capa de aplanamiento

cnn1.add(
    keras.layers.Flatten()
)

# Definición de capa totalmente conectada

cnn1.add(
    keras.layers.Dense(
        units = 128,
        activation = 'relu'
    )
)

# Definición de capa de salida

cnn1.add(
    keras.layers.Dense(
        units = 1,
        activation = 'sigmoid'
    )    
)

cnn1.summary()

cnn1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name='recall'),
        metrics.Precision(name='precision'),
        metrics.AUC(name='auc')
        # F1 score personalizado si es necesario
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
                epochs = 20,
                validation_data = (x_test, y_test)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_precision, test_auc = cnn1.evaluate(x_test, y_test)

# Gráfica de Recall
plt.plot(history1.history['recall'], label='Recall en el entrenamiento')
plt.plot(history1.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Gráfica de Precisión
plt.plot(history1.history['precision'], label='Precisión en el entrenamiento')
plt.plot(history1.history['val_precision'], label='Precisión en la validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
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
y_pred_probs = cnn1.predict(x_test)
y_pred = np.round(y_pred_probs).astype(int)  # Redondear para obtener 0 o 1

# Generar la matriz de confusión
cm1 = confusion_matrix(y_test, y_pred)
# Visualizar la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

report = classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo'])
print("Reporte de Clasificación:")
print(report)

# ------------------------------------------ CNN2 ------------------------------------------ #

cnn2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name='recall'),
        metrics.Precision(name='precision'),
        metrics.AUC(name='auc')
        # F1 score personalizado si es necesario
    ]
)

history2 = cnn2.fit(
                x_train,
                y_train,
                batch_size=100,
                epochs = 10,
                validation_data = (x_test, y_test)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_precision, test_auc = cnn2.evaluate(x_test, y_test)

cnn2.summary()

# Gráfica de Recall
plt.plot(history2.history['recall'], label='Recall en el entrenamiento')
plt.plot(history2.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Gráfica de Precisión
plt.plot(history2.history['precision'], label='Precisión en el entrenamiento')
plt.plot(history2.history['val_precision'], label='Precisión en la validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
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

# -------------------------------- Dropout y regularización -------------------------------- #

#######probar una red con regulzarización L2
reg_strength = 0.001

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.1  

# ------------------------------------------ CNN3 ------------------------------------------ #

cnn3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn3.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name='recall'),
        metrics.Precision(name='precision'),
        metrics.AUC(name='auc')
        # F1 score personalizado si es necesario
    ]
)

history3 = cnn3.fit(
                x_train,
                y_train,
                batch_size=100,
                epochs = 10,
                validation_data = (x_val, y_val)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_recall, test_precision, test_auc = cnn3.evaluate(x_val, y_val)

cnn3.summary()

# Gráfica de Recall
plt.plot(history3.history['recall'], label='Recall en el entrenamiento')
plt.plot(history3.history['val_recall'], label='Recall en la validación')
plt.title('Recall durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Gráfica de Precisión
plt.plot(history3.history['precision'], label='Precisión en el entrenamiento')
plt.plot(history3.history['val_precision'], label='Precisión en la validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Gráfica de AUC
plt.plot(history3.history['auc'], label='AUC en el entrenamiento')
plt.plot(history3.history['val_auc'], label='AUC en la validación')
plt.title('AUC durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Gráfica de Pérdida
plt.plot(history3.history['loss'], label='Pérdida en el entrenamiento')
plt.plot(history3.history['val_loss'], label='Pérdida en la validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

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
        metrics.Precision(name='precision'),
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

# Gráfica de Precisión
plt.plot(history4.history['precision'], label='Precisión en el entrenamiento')
plt.plot(history4.history['val_precision'], label='Precisión en la validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
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
