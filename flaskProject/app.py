from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('asd.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    sum = int(request.form.get('1', False))+int(request.form.get('2', False))+int(request.form.get('3', False))+int(request.form.get('4', False))+int(request.form.get('5', False))+int(request.form.get('6', False))+int(request.form.get('7', False))+int(request.form.get('8', False))+int(request.form.get('9', False))+int(request.form.get('10', False))
    age = float(request.form.get('a', False))
    gender = request.form.get('g', False)
    ethnicity = request.form.get('e', False)
    jaundice = request.form.get('j', False)
    autism = request.form.get('au', False)
    relation = request.form.get('r', False)
    arr = np.array([[age, gender, ethnicity, jaundice, autism, sum, 0, relation]])
    pred = model.predict(arr)
    # print(age)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
