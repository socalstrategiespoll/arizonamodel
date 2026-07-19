import os
import sys
import time

import az_bayesian_model as model
import az_maricopa_feed as maricopa
import az_pima_feed as pima
import az_live_feed as live_feed
import az_publish as publish

GIST_ID = os.environ.get("GIST_ID")
GIST_TOKEN = os.environ.get("GIST_TOKEN")

CYCLES_PER_RUN = 2
SECONDS_BETWEEN_CYCLES = 120


def run_one_cycle():
    try:
        maricopa.update_model_from_maricopa()
        print("[Maricopa] updated OK")
    except Exception as e:
        print("[Maricopa] FAILED (may just mean file not posted yet):", e)

    try:
        pima.update_model_from_pima()
        print("[Pima] updated OK")
    except Exception as e:
        print("[Pima] FAILED (may just mean file not posted yet):", e)

    try:
        county_totals = live_feed.poll_once()
        print(f"[SOS/statewide] updated OK ({len(county_totals)} counties)")
    except Exception as e:
        print("[SOS/statewide] FAILED (may just mean feed not live yet):", e)

    try:
        snap = publish.publish_snapshot(GIST_ID, GIST_TOKEN)
        pct = snap["statewide"]["pctIn"]
        pBiggs = snap["statewide"]["pBiggs"]
        print(f"[Publish] OK — {pct:.1%} of vote in, P(Biggs)={pBiggs:.1f}%")
        return True
    except Exception as e:
        print("[Publish] FAILED:", e)
        return False


def main():
    if not GIST_ID or not GIST_TOKEN:
        print("Missing GIST_ID or GIST_TOKEN environment variables.")
        sys.exit(1)

    any_publish_failed = False
    for i in range(CYCLES_PER_RUN):
        print(f"\n--- Cycle {i + 1} of {CYCLES_PER_RUN} ---")
        ok = run_one_cycle()
        if not ok:
            any_publish_failed = True
        if i < CYCLES_PER_RUN - 1:
            print(f"Waiting {SECONDS_BETWEEN_CYCLES}s before next cycle...")
            time.sleep(SECONDS_BETWEEN_CYCLES)

    if any_publish_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

