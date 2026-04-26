import os
import sys
import base64
import cv2
import numpy as np
import sympy
import subprocess
import shutil
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_math', methods=['POST'])
def process_math():
    data = request.json
    image_data = data.get('image_data')
    step_by_step = data.get('step_by_step', False)

    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. HARD THRESHOLD (Prag fix la 200). 
    # Fără Otsu! Scapă de purici și citește perfect roșul tău ca fiind negru solid.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Îngroșăm liniile ca să "sudăm" eventualele crăpături din cifre
    kernel = np.ones((5,5), np.uint8) 
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Ignorăm punctele mici și gunoaiele (cum este punctul din poza ta de sub "+")
        if w > 20 and h > 20:
            rects.append([x, y, x+w, y+h])
            
    if not rects:
        return jsonify({"success": False, "message": "Nu am găsit scris valid."})
        
    # Sortăm de la stânga la dreapta
    rects.sort(key=lambda b: b[0])
    
    # 3. ÎMBINARE INTELIGENTĂ
    merged_rects = []
    for r in rects:
        if not merged_rects:
            merged_rects.append(r)
        else:
            last_rect = merged_rects[-1]
            overlap_x = min(last_rect[2], r[2]) - max(last_rect[0], r[0])
            width_last = last_rect[2] - last_rect[0]
            width_r = r[2] - r[0]
            
            # Unim bucățile DOAR dacă se suprapun cu cel puțin 30% pe orizontală (cum e la "=")
            if overlap_x > 0.3 * min(width_last, width_r):
                last_rect[0] = min(last_rect[0], r[0])
                last_rect[1] = min(last_rect[1], r[1])
                last_rect[2] = max(last_rect[2], r[2])
                last_rect[3] = max(last_rect[3], r[3])
            else:
                merged_rects.append(r)
    
    temp_dir = "temp_chars"
    os.makedirs(temp_dir, exist_ok=True)
    full_string = ""
    
    for i, (x_min, y_min, x_max, y_max) in enumerate(merged_rects):
        w = x_max - x_min
        h = y_max - y_min
        ink_roi = thresh[y_min:y_max, x_min:x_max]
        
        # Facem un pătrat cu un padding FOARTE MIC ca să păstrăm forma originală a literei
        max_dim = max(h, w)
        new_size = int(max_dim * 1.1)
        
        square = np.zeros((new_size, new_size), dtype=np.uint8)
        x_offset = (new_size - w) // 2
        y_offset = (new_size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = ink_roi
        
        resized = cv2.resize(square, (45, 45), interpolation=cv2.INTER_AREA)
        final_char_image = cv2.bitwise_not(resized)
        
        char_path = f"{temp_dir}/char_{i}.png"
        cv2.imwrite(char_path, final_char_image)
        
        try:
            script_path = os.path.join("Mathematical-Handwriting-recognition", "main.py")
            result = subprocess.run(
                [sys.executable, script_path, char_path],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                if output_lines:
                    recognized = output_lines[0].split(":")[-1].strip()
                    if recognized == 'times': recognized = '*'
                    elif recognized == 'div': recognized = '/'
                    elif recognized == 'pm': recognized = '+'
                    full_string += recognized
        except Exception as e:
            print(f"Eroare AI la caracterul {i}: {e}")
            
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not full_string:
         return jsonify({"success": False, "message": "Nu am putut recunoaște caracterele."})

    equation_text = full_string.replace('=', '').replace(' ', '')
    
    try:
        expr = sympy.sympify(equation_text)
        answer = sympy.solve(expr) if 'x' in equation_text.lower() else sympy.simplify(expr)
        
        response_data = {
            "success": True,
            "equation": full_string,
            "answer": str(answer),
            "steps": ""
        }
        
        if step_by_step:
            steps = f"1. AI a citit contururile: {full_string}\n"
            steps += f"2. Expresia curățată: {equation_text}\n"
            steps += f"3. Rezultatul matematic: {answer}"
            response_data["steps"] = steps
            
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"success": False, "message": f"AI a citit '{full_string}', dar SymPy a dat eroare de sintaxă."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)