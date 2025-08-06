import pickle
import random
import json

# Load the json intents

with open("intents.json", "r") as file:
    intents = json.load(file)

# Load the model and vectorizer from train.py

saveFile = open("model.pkl", "rb")
model = pickle.load(saveFile)

vec = open("vectorizer.pkl", "rb")
vectorizer = pickle.load(vec)

# Prompt user to input message infinitely, exiting if they input quit / exit

while True:
    text = input("Enter your message: ")
    if text == "quit" or text == "exit":
        break

    # Turn the input into a numeric vector

    numeric = vectorizer.transform([text])

    # Use the numeric vector to predict with the model

    prediction = model.predict_proba(numeric)[0]
    highest = max(prediction)
    idx = prediction.tolist().index(highest)

    # ^ Get the highest confident answer and compare it to a confidence threshold

    if highest < 0.55:
        print("Sorry, could you repeat that?")
    else:

        # Output a random response

        tag = model.classes_[idx]
        responses = intents[tag]["responses"]
        print(random.choice(responses))