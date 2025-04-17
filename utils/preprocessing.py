import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import Config

def load_dataset(directory):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Etiketi dosya adından çıkar (son 2 karakter .png'den önce)
            label = int(filename.split('.')[-2][-2:])
            
            # Resmi yükle ve ön işleme
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, Config.IMG_SIZE)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Kanal boyutu ekle
            
            images.append(img)
            labels.append(label)
    
    # Etiketleri one-hot encoding'e dönüştür
    labels = to_categorical(np.array(labels) - 1, num_classes=Config.NUM_CLASSES)
    return np.array(images), labels

def prepare_data():
    X_train, y_train = load_dataset(Config.TRAIN_DIR)
    X_test, y_test = load_dataset(Config.TEST_DIR)
    return X_train, y_train, X_test, y_test