from flask import Flask, request, jsonify, render_template
import joblib
import openai
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load knowledge base
with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()


app = Flask(__name__)

# Load the trained model
model = joblib.load('lifestyle_model.pkl')

@app.route('/')
def welcome():
    return render_template('a.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/article.html')
def article():
    return render_template('article.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
        age = data['age']
        calories = data['calories']
        sleep = data['sleep']
        income = data['income']
        working_hours = data['working_hours']

        # Make prediction using the loaded model
        prediction = model.predict([[age, calories, sleep, income, working_hours]])

        # Prepare response
        response = {'prediction': prediction[0]}
    except Exception as e:
        response = {'error': str(e)}

    return jsonify(response)
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')

        messages = [
            {"role": "system", "content": f"You are a helpful medical assistant. Use the following knowledge base:\n{knowledge}"},
            {"role": "user", "content": user_input}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
