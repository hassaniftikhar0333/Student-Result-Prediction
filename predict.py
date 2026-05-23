import joblib

# Load trained model
model = joblib.load("student_model.pkl")

print("Model Loaded Successfully!")

# New student data
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))

# Prediction
prediction = model.predict([[hours, attendance]])

# Result
if prediction[0] == 1:
    print("Student Will PASS")
else:
    print("Student Will FAIL")