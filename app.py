import json
import time

import ollama
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


OLLAMA_MODEL = "llama3.2-vision"

print(f"SolveX Air pornit. Folosim Ollama (Model: {OLLAMA_MODEL})")


@app.route("/")
def index():
    return render_template("index.html")


def extract_base64_image(image_data):
    if not image_data or "," not in image_data:
        raise ValueError("Nu am primit o imagine valida.")
    return image_data.split(",", 1)[1]


def read_ollama_json(raw_text):
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start:end + 1])
        raise


def normalize_steps(steps):
    if isinstance(steps, list):
        return "\n".join(str(step) for step in steps)
    return steps or ""


def build_prompt(step_by_step):
    if step_by_step:
        return """
Esti profesor de matematica. Citeste ecuatia din imagine si rezolv-o.
Raspunde DOAR cu JSON valid, fara markdown:
{
  "equation": "ecuatia citita",
  "answer": "rezultatul final",
  "steps": ["1. pas complet", "2. pas complet", "3. pas complet"]
}
Scrie 3-6 pasi completi, clari si scurti.
Nu folosi niciodata "...", "[...]", "etc." sau rezumate incomplete.
Fiecare pas trebuie sa fie o propozitie completa.
"""

    return """
Esti un calculator matematic rapid. Citeste ecuatia din imagine si rezolv-o.
Raspunde DOAR cu JSON valid, fara markdown:
{
  "equation": "ecuatia citita",
  "answer": "rezultatul final",
  "steps": ""
}
Nu explica pasii.
"""


@app.route("/process_math", methods=["POST"])
def process_math():
    started_at = time.perf_counter()

    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image_data")
        step_by_step = bool(data.get("step_by_step", False))
        encoded_image = extract_base64_image(image_data)

        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=build_prompt(step_by_step),
            images=[encoded_image],
            format="json",
            keep_alive="10m",
            options={
                "temperature": 0,
                "num_predict": 900 if step_by_step else 160,
                "num_ctx": 4096,
            },
        )

        ai_data = read_ollama_json(response["response"])

        return jsonify(
            {
                "success": True,
                "equation": ai_data.get("equation", ""),
                "answer": ai_data.get("answer", ""),
                "steps": normalize_steps(ai_data.get("steps", "")),
                "is_explain": step_by_step,
                "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            }
        )

    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
