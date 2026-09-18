import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

SRC = Path(__file__).resolve().parent.parent / "data" / "features.csv"
DIVISIONS = ["Div1", "Div2", "Div3", "Div4", "Div1+2", "Global", "ICPC"]


def load():
    with SRC.open(newline="") as f:
        return list(csv.DictReader(f))


def build_matrix(rows):
    get = lambda key: np.array([float(r[key] or 0) for r in rows])
    old, field = get("old_rating"), get("field_avg_rating")
    participants, prior_count = get("num_participants"), get("prior_contest_count")
    y = get("rating_change")

    is_debut = (prior_count == 0).astype(float)
    div_dummies = np.array([[1.0 if r["division"] == d else 0.0 for d in DIVISIONS] for r in rows])

    X = np.column_stack([
        old - field, field, np.log(participants), np.log1p(prior_count), is_debut,
        get("prior_avg_rating_change"), get("prior_rating_change_std"), get("prior_best_rank"),
        get("prior_avg_rank"), get("days_since_last_contest"), get("rating_trend_last3"), div_dummies,
    ])
    return X, y


def chronological_split(contest_ids, train_frac=0.8, val_frac=0.1):
    ids = np.asarray(contest_ids)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    n = len(ids)
    train_end = int(starts[np.searchsorted(starts, int(train_frac * n))])
    val_end = int(starts[np.searchsorted(starts, int((train_frac + val_frac) * n))])
    return train_end, val_end


def main():
    rows = load()
    X, y = build_matrix(rows)
    train_end, val_end = chronological_split([int(r["contest_id"]) for r in rows])
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]

    model = LinearRegression()
    model.fit(X_train, y_train)

    for label, Xs, ys in (("train", X_train, y_train), ("val", X_val, y_val)):
        err = model.predict(Xs) - ys
        print(f"{label}: MAE {np.abs(err).mean():.2f}, RMSE {np.sqrt((err ** 2).mean()):.2f}")


if __name__ == "__main__":
    main()
