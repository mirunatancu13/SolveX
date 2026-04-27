import ast
import json
import math
import operator
import re
import time

import ollama
from flask import Flask, jsonify, render_template, request


try:
    import sympy as sp
except ImportError:
    sp = None

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


def build_ocr_prompt():
    return """
Citeste expresia sau ecuatia matematica din imagine.
Raspunde DOAR cu JSON valid, fara markdown:
{
  "equation": "ecuatia citita",
  "confidence": "high|medium|low"
}
Nu rezolva ecuatia. Nu explica. Scrie ecuatia cat mai compact.
Exemple: "2+3*4", "log_2 8", "2x+3=7".
"""


def build_fallback_prompt():
    return """
Esti profesor de matematica. Citeste ecuatia din imagine si rezolv-o.
Raspunde DOAR cu JSON valid, fara markdown:
{
  "equation": "ecuatia citita",
  "answer": "rezultatul final",
  "steps": ["1. pas complet", "2. pas complet", "3. pas complet"]
}
Scrie 5-8 pasi completi si clari, cu explicatii de cate 1-2 propozitii.
Nu folosi niciodata "...", "[...]", "etc." sau rezumate incomplete.
"""


def clean_equation(equation):
    equation = str(equation or "").strip()
    replacements = {
        "\u00d7": "*",
        "\u00b7": "*",
        "\u00f7": "/",
        "\u2212": "-",
        "^": "**",
    }
    for old, new in replacements.items():
        equation = equation.replace(old, new)
    return re.sub(r"\s+", " ", equation)


def format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


ALLOWED_AST_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_expression(expression):
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_AST_OPERATORS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Putere prea mare pentru evaluare locala.")
            return ALLOWED_AST_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_AST_OPERATORS:
            return ALLOWED_AST_OPERATORS[type(node.op)](eval_node(node.operand))
        raise ValueError("Expresie prea complexa pentru solverul local.")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed)


def solve_log_expression(equation):
    match = re.fullmatch(r"log_?(\d+(?:\.\d+)?)\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?", equation)
    if not match:
        return None

    base = float(match.group(1))
    value = float(match.group(2))
    answer = math.log(value, base)
    answer_text = format_number(round(answer, 10))
    base_text = format_number(base)
    value_text = format_number(value)
    return {
        "answer": answer_text,
        "steps": [
            f"1. Identificam expresia ca fiind logaritm: log in baza {base_text} din {value_text}. Un logaritm intreaba la ce putere trebuie ridicata baza ca sa obtinem numarul dat.",
            f"2. Rescriem ideea sub forma exponentiala: daca rezultatul este x, atunci {base_text}^x = {value_text}. Aceasta forma este de obicei mai usor de verificat.",
            f"3. Cautam o putere cunoscuta a lui {base_text}. Observam ca {base_text}^{answer_text} = {value_text}.",
            f"4. Pentru ca baza {base_text} ridicata la puterea {answer_text} da exact {value_text}, exponentul cautat este {answer_text}.",
            f"5. Deci valoarea finala a expresiei log_{base_text}({value_text}) este {answer_text}.",
        ],
    }


def solve_with_sympy(equation):
    if sp is None:
        return None

    normalized = equation.replace(" ", "")
    normalized = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", normalized)
    x = sp.symbols("x")

    if "=" in normalized:
        left, right = normalized.split("=", 1)
        left_expr = sp.sympify(left)
        right_expr = sp.sympify(right)
        solutions = sp.solve(sp.Eq(left_expr, right_expr), x)
        if solutions:
            answer = f"x = {solutions[0]}"
            return {
                "answer": answer,
                "steps": [
                    f"1. Pornim de la ecuatia {equation}. Scopul este sa izolam necunoscuta pe o parte a egalului.",
                    "2. Mutam termenii fara x pe partea cealalta a egalului, folosind operatia inversa. Daca un termen este adunat, il scadem din ambele parti.",
                    "3. Dupa ce strangem termenii asemenea, ramane un termen de forma a*x = b. Asta inseamna ca x este inmultit cu un coeficient.",
                    "4. Impartim ambele parti la coeficientul lui x pentru a lasa necunoscuta singura.",
                    f"5. Solutia obtinuta este {answer}. Putem verifica inlocuind x in ecuatia initiala.",
                ],
            }

    expr = sp.sympify(normalized)
    result = sp.simplify(expr)
    return {
        "answer": str(result),
        "steps": [
            f"1. Pornim de la expresia {equation}. Mai intai o citim exact asa cum este scrisa.",
            "2. Aplicam ordinea operatiilor: intai parantezele, apoi puterile, apoi inmultirile si impartirile, iar la final adunarile si scaderile.",
            "3. Calculam fiecare parte in ordinea corecta, ca sa nu schimbam valoarea expresiei.",
            "4. Simplificam rezultatul intermediar pana cand nu mai raman operatii de facut.",
            f"5. Rezultatul final este {result}.",
        ],
    }


def solve_locally(equation):
    equation = clean_equation(equation)
    if not equation:
        return None

    log_result = solve_log_expression(equation)
    if log_result:
        return log_result

    try:
        sympy_result = solve_with_sympy(equation)
        if sympy_result:
            return sympy_result
    except Exception:
        pass

    if "=" not in equation and re.fullmatch(r"[0-9+\-*/().\s]+", equation):
        try:
            answer = safe_eval_expression(equation)
            answer_text = format_number(answer)
            return {
                "answer": answer_text,
                "steps": [
                    f"1. Pornim de la expresia {equation}. O evaluam pas cu pas, fara sa schimbam ordinea termenilor inutil.",
                    "2. Verificam mai intai daca exista paranteze. Operatiile din paranteze au prioritate.",
                    "3. Apoi calculam puterile, daca exista, pentru ca acestea se fac inaintea inmultirilor si impartirilor.",
                    "4. Urmeaza inmultirile si impartirile, efectuate de la stanga la dreapta.",
                    "5. La final facem adunarile si scaderile, tot de la stanga la dreapta.",
                    f"6. Dupa simplificarea tuturor operatiilor, rezultatul final este {answer_text}.",
                ],
            }
        except Exception:
            pass

    return None


def solve_with_ollama_fallback(encoded_image):
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=build_fallback_prompt(),
        images=[encoded_image],
        format="json",
        keep_alive="10m",
        options={
            "temperature": 0,
            "num_predict": 1400,
            "num_ctx": 4096,
        },
    )
    return read_ollama_json(response["response"])


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
            prompt=build_ocr_prompt(),
            images=[encoded_image],
            format="json",
            keep_alive="10m",
            options={
                "temperature": 0,
                "num_predict": 90,
                "num_ctx": 1024,
            },
        )

        ai_data = read_ollama_json(response["response"])
        equation = clean_equation(ai_data.get("equation", ""))
        local_result = solve_locally(equation)

        if local_result:
            ai_data = {
                "equation": equation,
                "answer": local_result["answer"],
                "steps": local_result["steps"] if step_by_step else "",
                "source": "ollama_ocr_local_solver",
            }
        else:
            ai_data = solve_with_ollama_fallback(encoded_image)
            ai_data["source"] = "ollama_full_fallback"

        return jsonify(
            {
                "success": True,
                "equation": ai_data.get("equation", ""),
                "answer": ai_data.get("answer", ""),
                "steps": normalize_steps(ai_data.get("steps", "")),
                "is_explain": step_by_step,
                "source": ai_data.get("source", ""),
                "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            }
        )

    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
