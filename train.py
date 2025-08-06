import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

# Open the json file in read mode and parse its data to the variable intents

jsonFile = open("intents.json", "r")
intents = json.load(jsonFile)

# Create the pattern / responses list

X = []
y = []

# Populate X and y

for intent_name, intent_data in intents.items():
    for intent_pattern in intent_data['patterns']:
        X.append(intent_pattern)
        y.append(intent_name)

# Import vectorizer

vectorizer = TfidfVectorizer()

# Learn vocab and IDF from the training sentences

vectorizer.fit(X)

# Convert training sentences into numeric vectors

X_vectors = vectorizer.transform(X)

# Import model

model = LogisticRegression()
model.fit(X_vectors, y)

# Check model accuracy

print("Accuracy:", accuracy_score(y, model.predict(X_vectors)))

# Create save file and dump model into it using pickle

saveFile = open("model.pkl", "wb")
pickle.dump(model, saveFile)

vec = open("vectorizer.pkl", "wb")
pickle.dump(vectorizer, vec)