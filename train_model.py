import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.preprocessing import prepare_data
from config import Config

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*Config.IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    # Verileri yükle
    X_train, y_train, X_test, y_test = prepare_data()
    
    # Modeli oluştur
    model = build_model()
    model.summary()
    
    # Callback'ler
    callbacks = [
        ModelCheckpoint(Config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # Eğitim
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == '__main__':
    train()