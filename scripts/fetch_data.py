"""Download the raw public series used in Stage 4 into data/raw/ and write a manifest.

REAL DATA SOURCES (all free, no key):
  FRED  DEXUSEU   EUR/USD spot, USD per EUR (Federal Reserve H.10 noon buying rate, New York)
  FRED  SOFR      US secured overnight financing rate, percent
  FRED  EVZCLS    CBOE EUR/USD ETF Volatility Index close (30-day implied vol, percent)
  ECB   EST       euro short-term rate (EUR STR), volume-weighted trimmed mean, percent
  FRED  ECBESTRVOLWGTTRMDMNRT   the same EUR STR series as mirrored by FRED (cross-check only)

    python scripts/fetch_data.py [output_dir]      # default: data/raw

Downloaded files are untrusted data: this script only writes them to disk and checks that the body looks
like the CSV it asked for (not an HTML/XML error page). It never executes or imports anything it downloads.
Nothing about the user is sent; requests carry a generic User-Agent.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
SOURCES = {
    "fred_DEXUSEU.csv": (FRED + "DEXUSEU", "observation_date,DEXUSEU"),
    "fred_SOFR.csv": (FRED + "SOFR", "observation_date,SOFR"),
    "fred_EVZCLS.csv": (FRED + "EVZCLS", "observation_date,EVZCLS"),
    "ecb_ESTR.csv": (
        "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata&startPeriod=2019-10-01",
        "KEY,FREQ,BENCHMARK_ITEM",
    ),
    "fred_ECBESTR_crosscheck.csv": (FRED + "ECBESTRVOLWGTTRMDMNRT", "observation_date,ECBESTRVOLWGTTRMDMNRT"),
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fxvol-data-fetch", "Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{url}: HTTP {resp.status}")
        return resp.read()


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    manifest, failed = [], []
    for name, (url, header) in SOURCES.items():
        try:
            body = fetch(url)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            failed.append((name, str(e)))
            continue
        if not body.lstrip().startswith(header.encode()):
            failed.append((name, f"unexpected content (does not start with {header!r})"))
            continue
        (out / name).write_bytes(body)
        manifest.append(
            {
                "file": name,
                "url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "rows": body.count(b"\n") - 1,
            }
        )
        print(f"ok   {name:32s} {len(body):>9,d} bytes  {manifest[-1]['rows']:>6,d} rows")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, why in failed:
        print(f"FAIL {name}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "raw"))
