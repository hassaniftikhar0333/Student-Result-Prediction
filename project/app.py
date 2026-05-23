from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('student_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])

    prediction = model.predict([[hours, attendance]])

    if prediction[0] == 1:
        result = 'Student Will PASS'
    else:
        result = 'Student Will FAIL'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
