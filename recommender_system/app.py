from flask import Flask, request, jsonify
import joblib
import os

from .recommender import Recommender
from .search import Search

app: Flask = Flask(__name__)
app_dir = "recommender_system/"

# Load the Recommender Model
model_path = os.path.abspath(app_dir + "models/model.obj")
model: Recommender = joblib.load(model_path)

# Add Product ID
model.data["ID"] = model.data.index
model.data = model.data.sort_values(by=["ratings_5max"], ascending=False)

search_data = model.data.drop(["img_url", "discount_price", "ratings_5max"], axis=1).T.apply(lambda x: " ".join(list(map(str, x.values.tolist()))))
searcher = Search(search_data)

# Return all data
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return f"{model.data.to_html()}"
    else:
        return {"data": list(model.data.T.to_dict().values())}


# Return 'n' Products
@app.route("/products/<int:n>", methods=["GET", "POST"])
def get_products(n: int):
    n_data = model.data.sample(n)
    if request.method == "GET":
        return f"{n_data.to_html()}"
    else:
        return {"data": n_data.ID.values.tolist()}


# Return 'n' recommendations for 'id' Product
@app.route("/recommendations/<int:id>/<int:n>", methods=["GET", "POST"])
def get_recommendations(id: int, n: int):
    product = model.data.iloc[id, :-2]

    recommendations = model.get_k_recommendations(product, n)
    if request.method == "GET":
        return f"{recommendations.to_html()} <h1>This will return '{n}' recommendations where Product ID: '{id}'</h1>"
    else:
        return {"data": recommendations.ID.values.tolist()}


# Get Results based on given parameters
@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "GET":
        query =  str(request.args.get("query"))
        prods = searcher.search(query)
        return model.data.iloc[prods, :].to_html()


    # filter data
    query = str(request.json.get("query"))
    return {"data": searcher.search(query)}
