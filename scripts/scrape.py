"""Download contest.ratingChanges for every finished Codeforces contest."""
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
