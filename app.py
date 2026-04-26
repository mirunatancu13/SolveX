import os
import base64
import json
import requests
import pytesseract
from PIL import Image
import io
import traceback
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURARE ---
# Folosim ID-ul exact din poza ta
WOLFRAM_APP_ID = "WW77W8HHE9" 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_math', methods=['POST'])
def process_math():
    try:
        data = request.json
        image_data = data.get('image_data')
        step_by_step = data.get('step_by_step', False)

        # 1. OCR (Citirea textului)
        encoded_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(encoded_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Îmbunătățim imaginea pentru OCR (opțional dar recomandat)
        detected_equation = pytesseract.image_to_string(img, config='--psm 6').strip()
        print(f"DEBUG OCR: {detected_equation}")

        if not detected_equation:
            return jsonify({"success": False, "message": "Nu am detectat nimic. Scrie mai clar!"})

        # 2. CERERE CĂTRE WOLFRAM
        # Folosim endpoint-ul de query dar cu format minimalist pentru a evita erorile de protocol
        url = "http://api.wolframalpha.com/v2/query"
        params = {
            "appid": WOLFRAM_APP_ID,
            "input": detected_equation,
            "output": "json",
            "format": "plaintext",
        }

        if step_by_step:
            # Încercăm să activăm pașii dacă ID-ul permite
            params["podstate"] = "Step-by-step solution"
            params["input"] = f"solve {detected_equation}"

        response = requests.get(url, params=params)
        
        # Dacă Wolfram dă eroare de autentificare
        if response.status_code != 200:
            return jsonify({"success": False, "message": f"Eroare API Wolfram: {response.status_code}"})

        res_json = response.json()
        queryresult = res_json.get("queryresult", {})

        # 3. EXTRAGERE DATE
        final_answer = "N/A"
        steps_html = ""

        if queryresult.get("success"):
            pods = queryresult.get("pods", [])
            for pod in pods:
                # Căutăm rezultatul
                if pod["title"] in ["Result", "Solution", "Solutions", "Value"]:
                    final_answer = pod["subpods"][0]["plaintext"]
                
                # Căutăm pașii
                if "step" in pod["title"].lower():
                    for subpod in pod["subpods"]:
                        pasi_text = subpod.get("plaintext", "")
                        if pasi_text:
                            steps_html += f"<div>{pasi_text.replace('\n', '<br>')}</div>"
        else:
            return jsonify({"success": False, "message": "Wolfram nu a putut rezolva această ecuație."})

        return jsonify({
            "success": True,
            "equation": detected_equation,
            "answer": final_answer,
            "steps": steps_html if steps_html else "Pașii detaliați nu sunt disponibili pentru acest ID."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)