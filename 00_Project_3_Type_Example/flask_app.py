from flask import Flask, render_template, redirect, request
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle

# Create an instance of Flask
app = Flask(__name__)

# Now let's load our model ( note, we just use it )
with open('best_xgb_model.pickle', "rb") as f:
    model = pickle.load(f)

feature_names = model.get_booster().feature_names
print(f"\nOur model has the following features:\n{feature_names}\n")

# Route to render index.html template using data from Mongo
@app.route("/", methods=["GET", "POST"])
def home():
    output_message = ""

    if request.method == "POST":
        recency = float(request.form["recency"])
        frequency = float(request.form["frequency"])
        monetary = float(request.form["monetary"])
        time = float(request.form["time"])

        # data must be converted to df with matching feature names before predict
        data = pd.DataFrame(np.array([[recency, frequency, monetary, time]]), columns=feature_names)
        print("About to predict for:")
        print(data)
        result = model.predict(data)
        print(f"\nThe model predicted{result}!")
        if result == 1:
            output_message = "Nice, you will donate soon, thank you ^_^"
        else:
            output_message = "Please consider donating :-("
    
    return render_template("index.html", message = output_message)

if __name__ == "__main__":
    app.run()
