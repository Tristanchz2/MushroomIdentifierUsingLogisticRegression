# predictor.py
import numpy as np
import pandas as pd

class MushroomPredictor:
    def __init__(self, pipe, kept_features, category_choices, empirical_probs):
        self.pipe = pipe
        self.kept_features = kept_features
        self.category_choices = category_choices
        self.empirical_probs = empirical_probs

    def predict_from_partial(self, partial_features, n_samples=200, random_state=42):
        rng = np.random.default_rng(random_state)
        rows = []
        for _ in range(n_samples):
            row = {}
            for col in self.kept_features:
                v = partial_features.get(col, "")
                if v is not None and v != "":
                    row[col] = str(v)
                else:
                    cats = self.category_choices[col]
                    probs = np.array([self.empirical_probs[col].get(c, 0.0) for c in cats], dtype=float)
                    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
                    row[col] = rng.choice(cats, p=probs)
            rows.append(row)

        X_mc = pd.DataFrame(rows, columns=self.kept_features)
        probs = self.pipe.predict_proba(X_mc)[:, 1]
        return float(probs.mean())
