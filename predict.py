import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import Config
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

    def predict_from_pdf(self, pdf_path):
        from pdf2image import convert_from_path
        images = convert_from_path("osmanli_belge.pdf", poppler_path=r"C:\Program Files\poppler-24.08.0\Library\bin")

        images = convert_from_path(pdf_path)
        results = []

        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            page_text = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 10:
                    char_img = gray[y:y+h, x:x+w]
                    try:
                        predicted_char = self.predict_character(char_img)
                        page_text.append(predicted_char)
                    except:
                        pass
            results.append(" ".join(page_text))
        return results

def advanced_pdf_ocr(pdf_path, tesseract_path=None):
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path("osmanli_belge.pdf", poppler_path=r"C:\Program Files\poppler-24.08.0\Library\bin")

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    ocr = OttomanOCR()
    images = convert_from_path(pdf_path)

    final_text = []
    for img in images:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)

        page_text = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 60:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                word_img = img_cv[y:y+h, x:x+w]

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
                page_text.append(word)

        final_text.append(" ".join(page_text))
    return final_text

if __name__ == "__main__":
    ocr = OttomanOCR()

    # 1. TEK KARAKTER TAHMİNİ
    try:
        test_img = cv2.imread("test_char3.png", cv2.IMREAD_GRAYSCALE)
        if test_img is not None:
            print(f"Tahmin edilen karakter: {ocr.predict_character(test_img)}")
    except Exception as e:
        print(f"⚠️ test_char.png okunamadı: {e}")

    # 2. PDF OCR + Tesseract + Model
    try:
        results = advanced_pdf_ocr("osmanli_belge.pdf", tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        for i, page in enumerate(results):
            print(f"\n📄 Sayfa {i+1}:\n{page}")
    except Exception as e:
        print(f"⚠️ PDF OCR hatası: {e}")
