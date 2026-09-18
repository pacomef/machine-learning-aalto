# Stage 1, Machine Learning project

Problem formulation:

The aim of this project is to find the expected rating gain from contestants in competitive programming contests. The website used to find data, Codeforces [1], works with an elo rating similar to chess, and each contest affects this rating. The goal is to determine the elo gain from this contest given the past performances of an individual. It's most certainly impossible to reach a perfect score in that kind of task, but I will try to make it work as much as possible. This is a supervised regression problem: the label is `rating_change = new_rating - old_rating`, i.e. how much elo a contestant gains or loses from that one contest.
The dataset consists of 11,917,755 datapoints that I collected, all representing a performance of some individual in a given contest. There are 974,594 accounts in total, and 1796 contests, ranging from 2010-02-19 to 2026-09-13.

The data is made of a lot of flags, showing for instance in which division is the contestant competing, and a lot of continuous data, like the current elo of the individual, etc. This will be detailed more later, but there are 18 features, among which 9 continuous ones, 8 binary ones, and an integer.

Methods : 

I first of all scrapped every contest I could find on codeforces.com, through their API [2]. There were 2,145 of them, but 349 had no rated participants, thus I just removed them and had 1,796 contests left. The API gave 7 out of the 18 features that are included, and the rest of them were constructed to make more sense of the data. This was especially important because I was planning on using Linear Regression as the first method, and I knew that adding new features, which give some non-linearity, was going to be useful to get better results.

What also matters a lot in determining the next performance in an upcoming contests is how many contests the individual took part in, because codeforces gives a serious boost to the first few contests. Accounts created since 2020 have a mean of +400, +270, +190, +100, +65, +25 of elo gain over the first 6 contests.

Feature selection:

It used to work the other way before 2020 though: accounts started at 1500 elo directly and actually tended to lose some on their first contest instead (around -60 on average). This convinced me that whether an account is at its very first contest matters a lot on its own, on top of just how many contests it played before, so I made a binary flag for that (is_debut) and kept it alongside the contest count. For the 8% of rows that are an actual debut, all the "past contest" features obviously don't exist yet, so I just set them to 0, which combined with the flag doesn't bias anything since multiplying by 0 cancels out whatever the model would have done with that feature.

To check that the features I picked actually make sense and not just guesses, I looked at how each one correlates with the elo change on the training rows. The strongest ones are the elo relative to the field average (-0.58), the debut flag (+0.47) and the log of the past contest count (-0.46). Some barely correlate at all, like the number of days since the last contest (-0.04) or the average past rank (-0.03), but I kept them anyway since correlation only looks at one feature at a time and won't catch interactions between features.

| feature | corr. | feature | corr. |
|---|---|---|---|
| rating_vs_field | -0.58 | prior_best_rank | +0.12 |
| is_debut | +0.47 | division_Div4 | +0.13 |
| log1p_prior_contest_count | -0.46 | division_Div3 | +0.10 |
| prior_avg_rating_change | +0.37 | division_Div2 | -0.10 |
| prior_rating_change_std | -0.34 | division_Div1 | -0.05 |
| field_avg_rating | -0.27 | days_since_last_contest | -0.04 |
| rating_trend_last3 | +0.29 | prior_avg_rank | -0.03 |
| log_num_participants | +0.22 | division_Global | -0.01 |
| | | division_Div1+2 | -0.01 |
| | | division_ICPC | -0.01 |

Something I made sure NOT to use as a feature is the rank obtained in the contest itself, or the new rating. Codeforces computes the elo change almost directly from the rank, so giving that to the model would basically be handing it the answer instead of making it predict anything.

Model:

I went with Linear Regression to start with. It predicts the elo change as a linear function of the 18 features:

    h(x) = w^T x + b = sum_j w_j x_j + b

where x is a data point's feature vector and w, b are learned from the training data. One of those 18 features is the elo relative to the field, `x_field = old_rating - field_avg_rating`. When I plotted the elo change against it, the relationship looked roughly linear (slope around -0.18), the weights stay easy to read afterwards, and fitting it is basically instant even on close to 10 million rows since it has a closed-form solution. I know it can't really capture the non-linear boost new accounts get though, so that's a limitation I'm aware of for this first method.

Loss function:

I trained it by minimizing the squared error, averaged over the n training points:

    L(w, b) = (1/n) * sum_i (y_i - h(x_i))^2

It's the standard loss for a regression problem like this one, and it has a closed-form solution, the normal equations, so there's no need to iterate:

    w* = (X^T X)^-1 X^T y

To judge how good it is I look at the mean absolute error, directly in elo points, and the root mean squared error.

Validation:

For splitting the data I didn't do it randomly. Codeforces changed its rating system over time (see the 2020 change above) and the whole population of players evolved a lot too, so a random split would basically let the model peek at the future while training. Instead I sorted everything chronologically and cut it into 80% training, 10% validation and 10% test, making sure each cut falls between two contests and never in the middle of one.

| set | rows | share | period |
|---|---|---|---|
| training | 9,536,412 | 80.0% | 2010-02-19 to 2025-01-12 |
| validation | 1,189,681 | 10.0% | 2025-01-17 to 2025-10-10 |
| test | 1,191,662 | 10.0% | 2025-10-12 to 2026-09-13 |

Right now the linear regression gets a validation MAE of 54.27, against 101.88 for just predicting the average elo gain every time, so it's already clearly doing something, even if there's obviously a hard limit to how well this can ever work, since a lot of what decides a performance just can't be known ahead of time.

| | MAE | RMSE |
|---|---|---|
| training | 56.38 | 73.38 |
| validation | 54.27 | 72.76 |
| baseline (predict training mean) | 101.88 | 147.06 |

Use of AI:

I used Claude to typeset this document, and also to debug a lot of the functions that I was using, whether it is for the scraping, the data processing, or the training. I wrote this document myself and asked Claude to correct mistakes. 

References:

[1] Codeforces. https://codeforces.com. Accessed 18 September 2026.

[2] Codeforces API Documentation. https://codeforces.com/apiHelp. Accessed 18 September 2026.

Appendix: code

scrape.py

```python
import json
import time
from pathlib import Path

import requests

API = "https://codeforces.com/api"
OUT = Path(__file__).resolve().parent.parent / "data" / "rating_changes"
OUT.mkdir(parents=True, exist_ok=True)


def api_get(method, params=None):
    for _ in range(3):
        try:
            data = requests.get(f"{API}/{method}", params=params, timeout=30).json()
            return data["result"] if data.get("status") == "OK" else None
        except Exception:
            time.sleep(5)
    return None


def main():
    contests = [c for c in api_get("contest.list", {"gym": "false"}) if c["phase"] == "FINISHED"]
    for c in sorted(contests, key=lambda c: c["id"]):
        path = OUT / f"{c['id']}.json"
        if path.exists():
            continue
        rc = api_get("contest.ratingChanges", {"contestId": c["id"]}) or []
        path.write_text(json.dumps(rc))
        time.sleep(2)


if __name__ == "__main__":
    main()
```

build_features.py

```python
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "rating_changes"
OUT = ROOT / "data" / "features.csv"

FIELDS = [
    "contest_id", "division", "handle", "time", "old_rating", "num_participants",
    "field_avg_rating", "prior_contest_count", "prior_avg_rating_change",
    "prior_rating_change_std", "prior_best_rank", "prior_avg_rank",
    "days_since_last_contest", "rating_trend_last3", "rating_change",
]


def division(contest_name):
    n = contest_name.lower()
    if "div. 1" in n and "div. 2" in n:
        return "Div1+2"
    for d in ("1", "2", "3", "4"):
        if f"div. {d}" in n:
            return f"Div{d}"
    if "educational" in n:
        return "Educational"
    if "global" in n:
        return "Global"
    if "icpc" in n:
        return "ICPC"
    return "Other"


def main():
    contests = []
    for p in RAW.glob("*.json"):
        rows = json.loads(p.read_text())
        if rows:
            contests.append((int(p.stem), rows))
    contests.sort(key=lambda t: t[1][0]["ratingUpdateTimeSeconds"])

    history = {}
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for contest_id, rows in contests:
            div = division(rows[0]["contestName"])
            field_avg = statistics.mean(r["oldRating"] for r in rows)
            for r in rows:
                h = history.setdefault(r["handle"], [])
                if h:
                    deltas = [new - old for _, old, new, _ in h]
                    ranks = [rank for *_, rank in h]
                    prior = {
                        "prior_contest_count": len(h),
                        "prior_avg_rating_change": statistics.mean(deltas),
                        "prior_rating_change_std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
                        "prior_best_rank": min(ranks),
                        "prior_avg_rank": statistics.mean(ranks),
                        "days_since_last_contest": (r["ratingUpdateTimeSeconds"] - h[-1][0]) / 86400,
                        "rating_trend_last3": sum(deltas[-3:]),
                    }
                else:
                    prior = {k: "" for k in FIELDS if k.startswith(("prior_", "days_", "rating_trend"))}
                    prior["prior_contest_count"] = 0
                writer.writerow({
                    **prior,
                    "contest_id": contest_id, "division": div, "handle": r["handle"],
                    "time": r["ratingUpdateTimeSeconds"], "old_rating": r["oldRating"],
                    "num_participants": len(rows), "field_avg_rating": round(field_avg, 1),
                    "rating_change": r["newRating"] - r["oldRating"],
                })
                h.append((r["ratingUpdateTimeSeconds"], r["oldRating"], r["newRating"], r["rank"]))


if __name__ == "__main__":
    main()
```

train.py

```python
import csv
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "data" / "features.csv"
DIVISIONS = ["Div1", "Div2", "Div3", "Div4", "Div1+2", "Global", "ICPC"]


def load():
    with SRC.open(newline="") as f:
        return list(csv.DictReader(f))


def build_matrix(rows):
    n = len(rows)
    get = lambda key: np.array([float(r[key] or 0) for r in rows])
    old, field = get("old_rating"), get("field_avg_rating")
    participants, prior_count = get("num_participants"), get("prior_contest_count")
    y = get("rating_change")

    is_debut = (prior_count == 0).astype(float)
    div_dummies = np.array([[1.0 if r["division"] == d else 0.0 for d in DIVISIONS] for r in rows])

    X = np.column_stack([
        np.ones(n), old - field, field, np.log(participants), np.log1p(prior_count), is_debut,
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

    beta = np.linalg.solve(X_train.T @ X_train + 1e-6 * np.eye(X_train.shape[1]), X_train.T @ y_train)

    for label, Xs, ys in (("train", X_train, y_train), ("val", X_val, y_val)):
        err = Xs @ beta - ys
        print(f"{label}: MAE {np.abs(err).mean():.2f}, RMSE {np.sqrt((err ** 2).mean()):.2f}")


if __name__ == "__main__":
    main()
```

