import math
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import Series

class Search:
    
    def __init__(self, df):
        self.reg = re.compile(r"\w+")
        self.vectorizer = CountVectorizer()
        self.df = self.vectorizer.fit_transform(df.values)
    
    def search(self, X):
        X = self.vectorizer.transform([X])
        res = Series([x[0] for x in cosine_similarity(self.df, X)])
        return list(res[res > 0].index)
