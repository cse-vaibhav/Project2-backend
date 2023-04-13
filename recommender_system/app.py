from flask import Flask, request
import joblib

# from .recommender import Recommender

app: Flask = Flask(__name__)
app_dir = "recommender_system/"

# Load the Recommender Model
model_path = app_dir + "/models/model.obj"
model = joblib.load(model_path)


@app.route("/", methods=["GET"])
def index():
    return "<h1>This will do nothing</h1>"


@app.route("/products/<int:n>", methods=["GET", "POST"])
def get_products(n: int):
    n_data = model.data.sample(n)
    if request.method == "GET":
        return f"{n_data.to_html()}"
    else:
        return list(n_data.drop("clusters", axis=1).T.to_dict().values())


@app.route("/recommendations/<int:id>/<int:n>", methods=["GET", "POST"])
def get_recommendations(id: int, n: int):
    product = model.data.iloc[id, :-1]
    recommendations = model.get_k_recommendations(product, n)
    if request.method == "GET":
        return f"{recommendations.to_html()} <h1>This will return '{n}' recommendations where Product ID: '{id}'</h1>"
    else:
        return list(recommendations.T.to_dict().values())
