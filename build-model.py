import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from recommender_system.recommender import Recommender

if __name__ == "__main__":

    data = pd.read_csv("data/final_data.csv", index_col=0)

    ohe_params = {
        "dtype":"int", 
        "sparse_output":False, 
        "handle_unknown":"ignore"
    }
    preprocessor = ColumnTransformer([    
        ("one_hot_encoding", OneHotEncoder(**ohe_params), [0, 6, 9]),
        ("scaling", MinMaxScaler(), [3, 7, 8]),
        ("ordinal", OrdinalEncoder(), [4, 5])
    ], remainder="drop")

    model = KMeans(n_clusters=11, n_init=10)
    recommender = Recommender(data.copy(), [], [
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    recommender.fit()
    print(recommender.make_clusters().head())

    joblib.dump(recommender, "recommender_system/models/model.obj")

