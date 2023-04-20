from flask import Flask, request
import joblib
import os
import json

from .recommender import Recommender

app: Flask = Flask(__name__)
app_dir = "recommender_system/"

# Load the Recommender Model
model_path = os.path.abspath(app_dir + "models/model.obj")
print(model_path)
model: Recommender = joblib.load(model_path)


@app.route("/", methods=["GET", "POST"])
def index():
    data_with_id = model.data.copy()
    data_with_id["ID"] = data_with_id.index
    if request.method == "GET":
        return f"{data_with_id.to_html()}"
    else:
        return {"data": list(data_with_id.head(10).T.to_dict().values())}


@app.route("/products/<int:n>", methods=["GET", "POST"])
def get_products(n: int):
    n_data = model.data.sample(n)
    if request.method == "GET":
        return f"{n_data.to_html()}"
    else:
        return {"data": list(n_data.drop("clusters", axis=1).T.to_dict().values())}


@app.route("/recommendations/<int:id>/<int:n>", methods=["GET", "POST"])
def get_recommendations(id: int, n: int):
    product = model.data.iloc[id, :-1]

    print("PRODUCTS", product)
    recommendations = model.get_k_recommendations(product, n)
    if request.method == "GET":
        return f"{recommendations.to_html()} <h1>This will return '{n}' recommendations where Product ID: '{id}'</h1>"
    else:
        return {"data": list(recommendations.T.to_dict().values())}
