import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import Config
import pytesseract
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Yolunu buraya yaz

class OttomanOCR:
    def __init__(self, model_path=Config.MODEL_SAVE_PATH):
        self.model = load_model(model_path)
        self.img_size = Config.IMG_SIZE
    
    def preprocess_image(self, image):
        if image is None:
            raise ValueError("Görsel bulunamadı veya açılamadı.")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.img_size)
        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=-1)
    
    def predict_character(self, image):
        processed = self.preprocess_image(image)
        pred = self.model.predict(np.array([processed]), verbose=0)
        predicted_class = np.argmax(pred) + 1
        return Config.LABEL_MAP[predicted_class]

def recognize_from_image(image_path):
    ocr = OttomanOCR()

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)

    full_text = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_img = img[y:y+h, x:x+w]

            gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            chars = []
            for cnt in contours:
                xc, yc, wc, hc = cv2.boundingRect(cnt)
                if wc > 3 and hc > 5:
                    char_img = gray[yc:yc+hc, xc:xc+wc]
                    try:
                        chars.append((xc, ocr.predict_character(char_img)))
                    except:
                        pass

            chars.sort(key=lambda x: x[0])
            word = "".join([c[1] for c in chars])
            full_text.append(word)

    return " ".join(full_text)

if __name__ == "__main__":
    try:
        result = recognize_from_image("sayfa1.png")  # Buraya görselinin adını yaz
        print("\n📄 Taranan Sayfa:")
        print(result)
    except Exception as e:
        print(f"⚠️ Görsel OCR hatası: {e}")
