from flask import Flask, request
import joblib

from recommender import Recommender

app = Flask(__name__)

# Load the Recommender Model
model_path = "models/model.obj"
model: Recommender = joblib.load(model_path)


@app.route("/", methods=["GET"])
def index():
    return "<h1>This will do nothing</h1>"


@app.route("/products/<int:n>", methods=["GET", "POST"])
def get_products(n: int):
    n_data = model.data.sample(n)
    if request.method == "GET":
        return f"{n_data.to_html()}"
    else:
        return str(n_data.to_json())


@app.route("/recommendations/<int:id>/<int:n>")
def get_recommendations(id: int, n: int):
    return f"<h1>This will return '{n}' recommendations where Product ID: '{id}'</h1>"


if __name__ == "__main__":
    app.run(debug=True)
