import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatbot_config import CHATBOT_NAME, SYSTEM_PROMPT, MODEL

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=api_key)

@app.get("/")
def index():
    return render_template("index.html", chatbot_name=CHATBOT_NAME)

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn't generate an answer. Please try a study-related question."
        return jsonify({"answer": answer})
    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "The chatbot could not process your request right now."
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
