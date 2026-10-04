"""Generate the parquet source for the on-demand feature demo (NB8)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "feast_repo_ondemand" / "data"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    users = [f"u_{i:03d}" for i in range(50)]
    now = pd.Timestamp.utcnow().tz_convert("UTC").floor("h")
    rows = []
    for u in users:
        base = float(rng.uniform(50_000, 5_000_000))   # VND, wide spread on purpose
        for days_ago in (2, 1, 0):
            rows.append({
                "user_id": u,
                "event_timestamp": now - pd.Timedelta(np.int64(days_ago), unit="D"),
                "avg_amount_7d": round(base * float(rng.uniform(0.9, 1.1)), 2),
                "txn_count_7d": int(rng.integers(3, 40)),
                "created": now - pd.Timedelta(np.int64(days_ago), unit="D"),
            })
    df = pd.DataFrame(rows)
    path = OUT / "user_spend.parquet"
    df.to_parquet(path, index=False)
    print(f"wrote {path} ({len(df)} rows, {df.user_id.nunique()} users)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
