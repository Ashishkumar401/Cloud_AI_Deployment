from flask import Flask, render_template, request
import os, joblib
import pandas as pd
from train_auto import auto_train

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/model.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        acc = auto_train(file_path)
        return render_template('train.html', accuracy=round(acc * 100, 2))
    else:
        return "Please upload a valid CSV file!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = [float(x) for x in request.form.values()]
        model = joblib.load(MODEL_PATH)
        prediction = model.predict([data])[0]
        return render_template('predict.html', result=str(prediction))
    return render_template('predict.html')


# ✅ IMPORTANT FIX for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
