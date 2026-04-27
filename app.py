import os
import base64
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Dacă ești pe Windows, verifică calea către tesseract.exe
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_ocr(image_bytes):
    # 1. Transformăm byte-urile în format OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Convertim în GrayScale (alb-negru)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Mărim imaginea (Resizing) - Tesseract citește mai bine caracterele mari
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 4. Binarizare (Thresholding) - transformăm tot ce e scris în negru pur pe fundal alb pur
    # Folosim OTSU pentru a calcula automat pragul optim de contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. Dilation (Îngroșarea liniilor) - ajută dacă ai scris cu un creier subțire
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 6. Invertim înapoi (Tesseract preferă text negru pe fundal alb)
    final_img = cv2.bitwise_not(thresh)
    
    return final_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_math', methods=['POST'])
def process_math():
    try:
        data = request.json
        image_data = data.get('image_data')
        encoded_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(encoded_data)

        processed_img = preprocess_for_ocr(image_bytes)

        # Folosim PSM 6 sau 7 și permitem citirea caracterelor matematice
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/() '
        detected_equation = pytesseract.image_to_string(processed_img, config=custom_config).strip()

        # FIX pentru erori comune de OCR:
        # Dacă Tesseract vede paranteze ca cifre, încercăm să curățăm
        detected_equation = "".join(detected_equation.split())
        
        # Validare paranteze: dacă avem paranteză închisă dar nu și deschisă
        if ")" in detected_equation and "(" not in detected_equation:
            # Dacă ecuația începe cu o cifră care pare a fi paranteză (ex: 3 în loc de ()
            if detected_equation[0] in "317": 
                detected_equation = "(" + detected_equation[1:]

        try:
            # Calculăm rezultatul
            result = eval(detected_equation, {"__builtins__": None}, {})
            return jsonify({
                "success": True,
                "equation": detected_equation,
                "answer": str(result),
                "steps": "Calculat cu succes local."
            })
        except:
            return jsonify({
                "success": False,
                "equation": detected_equation,
                "message": f"Ecuație invalidă: {detected_equation}"
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
if __name__ == '__main__':
    app.run(debug=True, port=5000)