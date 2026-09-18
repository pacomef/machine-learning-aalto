# Codeforces rating change predictor

CS-C3240 Machine Learning project. Predicts `rating_change = new_rating - old_rating`
for a Codeforces account's next rated contest, using only information known before
the contest starts.

## Usage

```
pip install requests numpy scikit-learn
python scripts/scrape.py          # downloads data/rating_changes/<contest_id>.json
python scripts/build_features.py  # writes data/features.csv
python scripts/train.py           # fits linear regression, prints MAE/RMSE
```

## Files

- `scripts/scrape.py` — downloads `contest.ratingChanges` for every finished contest from the Codeforces API
- `scripts/build_features.py` — turns the cached contest data into one row per (handle, contest), with features built only from each account's earlier contests
- `scripts/train.py` — fits `sklearn.linear_model.LinearRegression`, with a chronological 80/10/10 train/validation/test split
- `report/` — the project report (source and PDF)
