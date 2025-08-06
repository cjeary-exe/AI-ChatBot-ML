from flask import Flask, render_template, request
import pickle
import json
import random

# Initialize Flask app

app = Flask(__name__)

# Load models and json

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("intents.json") as file:
    intents = json.load(file)

# Create the homepage route

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["message"]
    numeric = vectorizer.transform([user_input])
    prediction = model.predict_proba(numeric)[0]
    highest = max(prediction)
    idx = prediction.tolist().index(highest)

    if highest < 0.55:
        bot_response = "I'm sorry, please could you rephrase that?"
    else:
        tag = model.classes_[idx]
        bot_response = random.choice(intents[tag]["responses"])

    return render_template("index.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)