import matplotlib.pyplot as plt

def plot_training_history(history):
    """Modelin eğitim ve doğrulama doğruluğunu/hatayı çizer."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_acc, 'ro-', label='Doğrulama Doğruluğu')
    plt.title('Doğruluk')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'ro-', label='Doğrulama Kaybı')
    plt.title('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()
