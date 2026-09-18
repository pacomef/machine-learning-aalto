"""Build one row per (handle, contest) rated appearance from the cached rating changes.

Only pre-contest information is written out: each account's prior_* columns
come from its earlier contests only, and the contest's own rank/new rating
are never included (they determine rating_change almost deterministically,
so using them as inputs would make prediction circular).
"""
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

    history = {}  # handle -> [(time, old_rating, new_rating, rank), ...]
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
