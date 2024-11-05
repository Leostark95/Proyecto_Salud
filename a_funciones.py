import numpy as np
import cv2

from os import listdir # para hacer lista de archivos en una ruta
from tqdm import tqdm  # para crear contador en un for para ver evolución
from os.path import join # para unir ruta con archivo 

def img2data(path, width=224):
    
    rawImgs = []   # una lista con el array que representa cada imágen
    labels = [] # el label de cada imágen
    
    list_labels = [path+f for f in listdir(path)]

    for imagePath in list_labels: # recorre cada carpeta de la ruta ingresada
        
        files_list=listdir(imagePath) # crea una lista con todos los archivos
        for item in tqdm(files_list): # le pone contador a la lista: tqdm
            file = join(imagePath, item) # crea ruta del archivo
            if file[-1] =='g': # verificar que se imágen extensión jpg o jpeg
                img = cv2.imread(file) # cargar archivo
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # invierte el orden de los colores en el array para usar el más estándar RGB
                img = cv2.resize(img ,(width,width)) # cambia resolución de imágnenes
                rawImgs.append(img) # adiciona imágen al array final
                l = imagePath.split('/')[2] # identificar en qué carpeta está
                if l == '0':  # verificar en qué carpeta está para asignar el label
                    labels.append([0])
                elif l == '1':
                    labels.append([1])
    return rawImgs, labels, files_list