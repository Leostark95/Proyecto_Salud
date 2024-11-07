import numpy as np
import cv2

from os import listdir # para hacer lista de archivos en una ruta
from tqdm import tqdm  # para crear contador en un for para ver evolución
from os.path import join # para unir ruta con archivo 
import matplotlib.pyplot as plt
import seaborn as sns

def img2data(path, width=224, colormap=cv2.COLORMAP_VIRIDIS):
    rawImgs = []  # lista con arrays que representan cada imagen
    labels = []   # lista de etiquetas de cada imagen
    
    list_labels = [join(path, f) for f in listdir(path)]  # rutas completas de las carpetas en path

    for imagePath in list_labels:  # recorre cada carpeta en la ruta ingresada
        files_list = listdir(imagePath)  # lista de archivos en la carpeta actual
        for item in tqdm(files_list):  # progreso de carga de archivos con tqdm
            file = join(imagePath, item)  # crea ruta completa del archivo
            if file.lower().endswith(('jpg', 'jpeg')):  # verificar si es una imagen (jpg o jpeg)
                img = cv2.imread(file)  # cargar archivo
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convertir a RGB
                img = cv2.resize(img, (width, width))  # cambiar resolución
                
                # Convertir a escala de grises
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Aplicar un umbral binario adaptativo
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 5, 2)

                # Aplicar operaciones morfológicas para limpiar el área de interés
                kernel = np.ones((5,5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # Aplicar la máscara al fondo
                img_colored_background = cv2.bitwise_and(img, img, mask=thresh)

                # Aplicar el colormap
                img_colormap = cv2.applyColorMap(img_colored_background, colormap)
                
                rawImgs.append(img_colormap)  # añadir imagen procesada al array final
                
                # Asignar etiquetas en función de la carpeta
                l = imagePath.split('/')[-1]  # extraer el nombre de la carpeta
                if l == '0':  # asignar etiqueta basada en el nombre de la carpeta
                    labels.append([0])
                elif l == '1':
                    labels.append([1])

    return rawImgs, labels, files_list

#Función para convertir a array
def imag_array():

    trainpath = 'data/train/'
    testpath = 'data/test/'
    valpath = 'data/valid/'

    x_train, y_train, file_list= img2data(trainpath)
    x_test, y_test, file_list= img2data(testpath)
    x_val, y_val, file_list= img2data(valpath)

    #### convertir salidas a numpy array ####
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    return x_train, y_train, x_test, y_test, x_val, y_val

# Gráfica de Recall
def plot_recall(history1):
    plt.plot(history1.history['recall'], label='Recall en el entrenamiento')
    plt.plot(history1.history['val_recall'], label='Recall en la validación')
    plt.title('Recall durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

    # Gráfica de AUC
def plot_auc(history1):
    plt.plot(history1.history['auc'], label='AUC en el entrenamiento')
    plt.plot(history1.history['val_auc'], label='AUC en la validación')
    plt.title('AUC durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    # Gráfica de Pérdida
def plot_perdida(history1):
    plt.plot(history1.history['loss'], label='Pérdida en el entrenamiento')
    plt.plot(history1.history['val_loss'], label='Pérdida en la validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

#Visualización matriz de confución
def matriz(cm1):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.show()