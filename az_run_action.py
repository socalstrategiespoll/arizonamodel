import os
import sys

import az_bayesian_model as model
import az_maricopa_feed as maricopa
import az_pima_feed as pima
import az_civicapi_feed as civicapi
import az_live_feed as live_feed
import az_publish as publish

GIST_ID = os.environ.get("GIST_ID")
GIST_TOKEN = os.environ.get("GIST_TOKEN")


def main():
    if not GIST_ID or not GIST_TOKEN:
        print("Missing GIST_ID or GIST_TOKEN environment variables.")
        sys.exit(1)

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
        updated = civicapi.update_model_from_civicapi()
        print(f"[civicAPI] updated OK ({len(updated)} counties: {updated})")
    except Exception as e:
        print("[civicAPI] FAILED, falling back to SOS feed:", e)
        try:
            county_totals = live_feed.poll_once()
            print(f"[SOS/statewide fallback] updated OK ({len(county_totals)} counties)")
        except Exception as e2:
            print("[SOS/statewide fallback] FAILED too (may just mean neither source is live yet):", e2)

    try:
        snap = publish.publish_snapshot(GIST_ID, GIST_TOKEN)
        pct = snap["statewide"]["pctIn"]
        pBiggs = snap["statewide"]["pBiggs"]
        print(f"[Publish] OK — {pct:.1%} of vote in, P(Biggs)={pBiggs:.1f}%")
    except Exception as e:
        print("[Publish] FAILED:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
