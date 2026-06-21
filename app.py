from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("food_model.h5")

# Load calorie data
data = pd.read_csv("calorie_data.csv")

# Classes used during training
classes = ["Apple", "Banana", "Burger", "Dosa", "Idli", "Pizza"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"


    # quantity from user
    quantity = int(request.form["quantity"])


    # save image
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)


    # image processing
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)


    # prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = round(np.max(prediction)*100,2)

    food = classes[predicted_class]


    # calories
    row = data[data["food"].str.lower() == food.lower()]


    if not row.empty:

        calories_per_item = row.iloc[0]["calories"]

        total_calories = calories_per_item * quantity

    else:

        total_calories = "Not Found"



    return render_template(
        "result.html",
        food=food,
        quantity=quantity,
        calories=total_calories,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)