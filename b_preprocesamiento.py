
# -------------------------------- Importanción de librerías ----------------------------------#

import pandas as pd
import numpy as np
import a_funciones as fn
import cv2
import joblib
import os
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from a_funciones import img2data
from imgaug import augmenters as iaa

# ---------------------------- Ejemplos de imágenes del dataset -------------------------------#

# Lectura de imagenes

img1 = cv2.imread('data/train/0/393_1317322195_png.rf.fa1093047a7ba1b8998ad64120b5a0d1.jpg')
img2 = cv2.imread('data/train/1/9851_1434430689_png.rf.3b1a584f0be68600f30ecb7c30d5938e.jpg')

# representación de imágenes

plt.imshow(img1)
plt.title('Diagnótico Negativo')
plt.show()

plt.imshow(img2)
plt.title('Diagnóstico Positivo')
plt.show()

# representación numérica de imágenes

img1.shape
img1.max()
img1.min()

img2.shape
img2.max()
img2.min()

# Cantidad de pixerles en la imagen alto * ancho * canales

np.prod(img1.shape)
np.prod(img2.shape)

#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes

img1_r = cv2.resize(img1 ,(100,100))
plt.imshow(img1_r)
plt.title('Diagnótico Negativo')
plt.show()
np.prod(img1_r.shape)

img2_r = cv2.resize(img2 ,(100,100))
plt.imshow(img2_r)
plt.title('Diagnótico Positivo')
plt.show()
np.prod(img2_r.shape)

# --------------------------------- Distribuciones de las clases ---------------------------- #

dataset_dir = 'data'

subsets = ["train", "test", "valid"]
classes = ["0", "1"]  # 0 para benigno, 1 para maligno

data_summary = {"Subset": [], "Class": [], "Image Count": []}

for subset in subsets:
    for class_label in classes:
        class_dir = os.path.join(dataset_dir, subset, class_label)
        image_count = len(os.listdir(class_dir))
        data_summary["Subset"].append(subset)
        data_summary["Class"].append(class_label)
        data_summary["Image Count"].append(image_count)

df_summary = pd.DataFrame(data_summary)

print("Resumen del Dataset:")
print(df_summary)

# ------------------------------------ Balance de las clases -------------------------------- #

total_images = df_summary.groupby("Subset")["Image Count"].sum().reset_index(name="Total Images")

df_summary = pd.merge(df_summary, total_images, on="Subset")

df_summary["Class Percentage"] = ((df_summary["Image Count"] / df_summary["Total Images"]) * 100).round(2)

print("Análisis Descriptivo del Balance de Clases:")
print(df_summary)

# ---------------------------------- Función img2data aplicada ------------------------------ #

rawImgs, labels, _ = img2data(dataset_dir + "/train/")

# ---------------------------------- Visualización de imágenes ------------------------------ #

num_images = 5
# Convertir la lista de imágenes y etiquetas a un DataFrame para manipularlas fácilmente
data = {"Image": rawImgs, "Label": [label[0] for label in labels]}

def plot_images(data, label, num_images):
    images_to_plot = [img for img, lbl in zip(data["Image"], data["Label"]) if lbl == label][:num_images]
    
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images_to_plot):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Benigno" if label == 0 else "Maligno")
    plt.show()

print("Muestras de Imágenes Benignas (Clase 0):")
plot_images(data, label=0, num_images=num_images)

print("Muestras de Imágenes Malignas (Clase 1):")
plot_images(data, label=1, num_images=num_images)

# ---------------------------------------- Normalización ------------------------------------ #

def normalize_images(images):
    # Convertir lista de imágenes en un arreglo numpy y escalar a rango [0, 1]
    images_normalized = np.array(images, dtype="float32") / 255.0
    return images_normalized

# Aplicar normalización
rawImgs_normalized = normalize_images(rawImgs)

# Confirmar si todas las imágenes tienen la misma dimensión
unique_sizes = set(img.shape for img in rawImgs_normalized)
if len(unique_sizes) == 1:
    print("\nTodas las imágenes normalizadas tienen las mismas dimensiones:", unique_sizes.pop())
else:
    print("\nLas imágenes tienen diferentes dimensiones:", unique_sizes)

# --------------------------------------- Aumento de datos ---------------------------------- #

# Definir una secuencia de aumento de datos

def augment_images(images):
    augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),                       # 50% de probabilidad de aplicar espejado horizontal
    iaa.Rotate((-15, 15)),                 # Rotar entre -15 y 15 grados
    iaa.Multiply((0.8, 1.2)),              # Ajuste de brillo entre 80% y 120%
    iaa.GaussianBlur(sigma=(0.0, 1.0))])   # Aplicar desenfoque gaussiano leve
    # Aplicar el aumento de datos a las imágenes
    images_augmented = augmentation_pipeline(images=images)
    return images_augmented

def plot_image_samples(images, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_image_samples(rawImgs, title="Imágenes Originales")

# Normalizar las imágenes
rawImgs_normalized = normalize_images(rawImgs)
plot_image_samples(rawImgs_normalized, title="Imágenes Normalizadas")

# Aplicar aumento de datos (data augmentation) solo al conjunto de entrenamiento
rawImgs_augmented = augment_images(rawImgs_normalized)
plot_image_samples(rawImgs_augmented, title="Imágenes con Aumento de Datos")

# -------------------------------------- Train, Test, Valid --------------------------------- #

# para cargar todas las imágenes
# reducir su tamaño y convertir en array

trainpath = 'data/train/'
testpath = 'data/test/'
valpath = 'data/valid/'

x_train, y_train, file_list= fn.img2data(trainpath)

# Normalizar las imágenes y actualizar x_train
x_train_norm = normalize_images(x_train)
plot_image_samples(x_train, title="Imágenes Normalizadas")

# Aplicar aumento de datos y actualizar x_train
x_train_aug = augment_images(x_train)
plot_image_samples(x_train, title="Imágenes con Aumento de Datos")

x_train = []

x_train.append(x_train_norm)
x_train.append(x_train_aug)

print(len(x_train_norm))






















x_test, y_test, file_list = fn.img2data(testpath)
x_val, y_val, file_list = fn.img2data(valpath) 

#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)

x_train.shape
x_test.shape
x_val.shape


np.prod(x_train[1].shape)
y_train.shape
y_test.shape
y_val.shape

# ----------------------- salidas del preprocesamiento bases listas ------------------------- #

joblib.dump(x_train, "salidas\\x_train.pkl")
joblib.dump(y_train, "salidas\\y_train.pkl")
joblib.dump(x_test, "salidas\\x_test.pkl")
joblib.dump(y_test, "salidas\\y_test.pkl")
joblib.dump(x_val, "salidas\\x_val.pkl")
joblib.dump(y_val, "salidas\\y_val.pkl")