import math
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class Search:
    
    def __init__(self, df, unique_values):
        self.reg = re.compile(r"\w+")
        self.vectorizer = CountVectorizer()
        self.df = self.vectorizer.fit_transform(df)
        self.unique_values = unique_values
        
    def validate_search(self, search_term):
        
        def jaccard_similarity(query, document):
            intersection = set(query).intersection(set(document))
            union = set(query).union(set(document))
            return len(intersection)/len(union)

        possible = []
        for v in self.unique_values:
            sim = jaccard_similarity(search_term, v)
            if sim >= 0.5:
                possible.append((sim, v))
        possible.sort(key=lambda x: x[0], reverse=True)
        print(possible)
        if len(possible) > 0:
            return possible[0][1]
        else:
            return None
    
    def search(self, X):
        X = self.validate_search(X)
        print(X)
        if not X:
            return []
        
        X = self.vectorizer.transform([X])
        res = pd.Series([x[0] for x in cosine_similarity(self.df, X)])
        return list(res[res > 0.2].index)