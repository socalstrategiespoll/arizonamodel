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

    maricopa_failed = False
    try:
        maricopa.update_model_from_maricopa()
        print("[Maricopa] updated OK")
    except Exception as e:
        maricopa_failed = True
        print("[Maricopa] FAILED (may just mean file not posted yet):", e)

    pima_failed = False
    try:
        pima.update_model_from_pima()
        print("[Pima] updated OK")
    except Exception as e:
        pima_failed = True
        print("[Pima] FAILED (may just mean file not posted yet):", e)

    try:
        skip = civicapi.HANDLED_BY_DEDICATED_FEED.copy()
        if pima_failed:
            skip.discard("Pima")
            print("[civicAPI] Pima's own feed failed — letting civicAPI cover Pima this cycle")
        if maricopa_failed:
            skip.discard("Maricopa")
            print("[civicAPI] Maricopa's own feed failed — letting civicAPI cover Maricopa this cycle")
        updated = civicapi.update_model_from_civicapi(skip_counties=skip)
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
