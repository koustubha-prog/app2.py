import numpy as np
from flask import Flask, request, render_template
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 4)
	loaded_model = pickle.load(open("model.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    to_predict_list = list(float_features)
    prediction = ValuePredictor(to_predict_list)
    return render_template("result.html", prediction_text = prediction)

if __name__ == "__main__":
    app.run(debug=True)
