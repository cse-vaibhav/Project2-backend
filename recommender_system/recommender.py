from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from typing import List, Optional


class Recommender(Pipeline):
    def __init__(self, data: pd.DataFrame, ignore_cols: List[str], steps: list) -> None:
        self.data = data  # copy of the entire data
        self.ignore_cols = ignore_cols  # columns to ignore during fit process
        super().__init__(steps)  # creating pipeline

    def fit(
        self,
        X: Optional[pd.DataFrame | np.ndarray] = None,
        y: Optional[pd.Series | np.ndarray] = None,
        **fit_params
    ):
        """Fit after dropping the ignore_cols"""

        data_fr: np.ndarray = self.drop_cols().values
        return super().fit(data_fr)

    def drop_cols(self) -> pd.DataFrame:
        """return data after dropping ignore_cols"""

        cols: List[str] = self.ignore_cols.copy()
        return self.data.drop(cols, axis=1)

    def make_clusters(self) -> pd.DataFrame:
        """
        Predict clusters and add it as "clusters" column in the data
        return the new data
        """

        self.data["clusters"] = self.predict(self.drop_cols().values)
        return self.data

    def get_cluster(self, user_pref: pd.DataFrame | pd.Series) -> int:
        """get the cluster a user's data belongs to"""

        if type(user_pref) is pd.Series:
            # Series.values returns a 1D array
            return super().predict([user_pref.values])[0]
        else:
            # DataFrame.values returns a 2D array
            return super().predict(user_pref.values)[0]

    def get_k_recommendations(self, user_pref, k=5) -> pd.DataFrame:
        """get 'k' recommendations based on user's data"""

        cluster: int = self.get_cluster(user_pref)
        return self.data[self.data["clusters"] == cluster].head(k)
