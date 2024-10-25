import numpy as np

import cv2 ### para leer imagenes jpeg
### pip install opencv-python

from matplotlib import pyplot as plt ## para gráfciar imágnes
import a_funciones as fn #### funciones personalizadas, carga de imágenes
import joblib ### para descargar array

############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

img1=cv2.imread('data/train/0/393_1317322195_png.rf.fa1093047a7ba1b8998ad64120b5a0d1.jpg')
# img2 = cv2.imread('data/train/PNEUMONIA/person7_bacteria_29.jpeg')


############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

plt.imshow(img1)
plt.title('normal')
plt.show()

plt.imshow(img2)
plt.title('pneumonia')
plt.show()

###### representación numérica de imágenes ####

img2.shape ### tamaño de imágenes
img1.shape
img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel

np.prod(img1.shape) ### 5 millones de observaciones cada imágen

#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes

img1_r = cv2.resize(img1 ,(100,100))
plt.imshow(img1_r)
plt.title('Normal')
plt.show()
np.prod(img1_r.shape)

img2_r = cv2.resize(img2 ,(100,100))
plt.imshow(img2_r)
plt.title('Normal')
plt.show()
np.prod(img2_r.shape)

################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################


width = 100
num_classes = 2 
trainpath = 'data/train/'
testpath = 'data/test/'
valpath = 'data/valid/'

x_train, y_train, file_list= fn.img2data(trainpath)
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

# Calcular la cantidad de imágenes por clase en cada conjunto
train_counts = np.unique(y_train, return_counts=True)
val_counts = np.unique(y_val, return_counts=True)
test_counts = np.unique(y_test, return_counts=True)

# Imprimir resultados
print("Distribución en el conjunto de entrenamiento:")
print(f"Clase 0 (Normal): {train_counts[1][0]}")
print(f"Clase 1 (Pneumonia): {train_counts[1][1]}")

print("\nDistribución en el conjunto de validación:")
print(f"Clase 0 (Normal): {val_counts[1][0]}")
print(f"Clase 1 (Pneumonia): {val_counts[1][1]}")

print("\nDistribución en el conjunto de prueba:")
print(f"Clase 0 (Normal): {test_counts[1][0]}")
print(f"Clase 1 (Pneumonia): {test_counts[1][1]}")