from flask import Flask, render_template, request
import joblib
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load ML model and vectorizer
model = joblib.load("model/news_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

app = Flask(__name__)

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return ' '.join([p.text for p in paragraphs if p.text])
    except Exception:
        return "Error extracting content from the URL."

# Function to get GPT prediction and explanation
def gpt_fact_check(statement):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI that detects if a news statement is fake or real."
                },
                {
                    "role": "user",
                    "content": f"Is the following news statement real or fake?\n\n\"{statement}\"\n\nRespond with 'Fake', 'Real', or 'Unverified' and then give a brief explanation."
                }
            ]
        )
        reply = response.choices[0].message.content.strip()
        if "fake" in reply.lower():
            return "Fake", reply
        elif "real" in reply.lower():
            return "Real", reply
        else:
            return "Unverified", reply
    except Exception as e:
        return "Unverified", f"GPT Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    verdict = ""
    input_text = ""
    url_input = ""
    ml_prediction = ""
    gpt_verdict = ""
    gpt_explanation = ""
    content_text = ""

    if request.method == "POST":
        input_text = request.form.get("statement", "").strip()
        url_input = request.form.get("url", "").strip()

        content_text = input_text or extract_text_from_url(url_input)

        # ML model prediction
        transformed = vectorizer.transform([content_text])
        pred = model.predict(transformed)[0]
        ml_prediction = "Real" if pred == 1 else "Fake"

        # GPT prediction
        gpt_verdict, gpt_explanation = gpt_fact_check(content_text)

        # Final verdict (e.g., you can customize logic here)
        verdict = gpt_verdict
        verdict_source = "GPT + ML"

        return render_template(
            "index.html",
            verdict=verdict,
            content_text=content_text,
            input_text=input_text,
            url_input=url_input,
            gpt_verdict=gpt_verdict,
            ml_prediction=ml_prediction,
            gpt_explanation=gpt_explanation,
            verdict_source=verdict_source
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
