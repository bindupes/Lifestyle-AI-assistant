from flask import Flask, request, jsonify, render_template
import joblib

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

if __name__ == '__main__':
    app.run(debug=True)
