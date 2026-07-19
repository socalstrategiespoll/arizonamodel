import re

import requests

import az_bayesian_model as model
import az_live_feed as live_feed

RACE_ID = 84359
RACE_URL = f"https://civicapi.org/api/v2/race/{RACE_ID}"

CANDIDATE_TO_KEY = {
    "BIGGS": "B",
    "SCHWEIKERT": "S",
}

COUNTY_NAMES = set(model.COUNTIES.keys())

HANDLED_BY_DEDICATED_FEED = {"Maricopa", "Pima"}


class CivicAPIError(Exception):
    pass


def fetch_race(race_id=RACE_ID, timeout=30):
    resp = requests.get(f"https://civicapi.org/api/v2/race/{race_id}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def diagnose_structure(data):
    print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  '{k}': list of {len(v)} items")
                if v:
                    print(f"    first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else v[0]}")
            elif isinstance(v, dict):
                print(f"  '{k}': dict with keys {list(v.keys())}")
            else:
                print(f"  '{k}': {v!r}")


def _candidate_key(name):
    name_upper = (name or "").upper()
    for surname, key in CANDIDATE_TO_KEY.items():
        if surname in name_upper:
            return key
    return "O"


def find_county_breakdown(data):
    region_results = data.get("region_results")
    if not isinstance(region_results, dict):
        return None

    county_totals = {}
    for slug, entry in region_results.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or slug.replace("_", " ").title()
        name_clean = name.strip()

        matched_name = None
        for known in COUNTY_NAMES:
            if known.lower() == name_clean.lower():
                matched_name = known
                break
        if matched_name is None:
            continue

        totals = {"B": 0, "S": 0, "O": 0}
        for c in entry.get("candidates", []):
            key = _candidate_key(c.get("name", ""))
            totals[key] += c.get("votes", 0) or 0
        county_totals[matched_name] = totals

    return county_totals if county_totals else None


def get_statewide_totals(data):
    totals = {"B": 0, "S": 0, "O": 0}
    for c in data.get("candidates", []):
        key = _candidate_key(c.get("name", ""))
        totals[key] += c.get("votes", 0) or 0
    return totals


FALLBACK_STATES = {
    "Maricopa": live_feed.LiveCountyState("Maricopa"),
    "Pima": live_feed.LiveCountyState("Pima"),
}


def update_model_from_civicapi(skip_counties=None):
    if skip_counties is None:
        skip_counties = HANDLED_BY_DEDICATED_FEED

    data = fetch_race()
    county_breakdown = find_county_breakdown(data)

    if county_breakdown is None:
        raise CivicAPIError(
            "civicAPI response has no recognizable county-level breakdown — "
            "only a statewide total is available. Call diagnose_structure() "
            "on the raw response to inspect it, or fall back to az_live_feed."
        )

    updated = []
    for county_name, totals in county_breakdown.items():
        if county_name in skip_counties:
            continue

        if county_name in FALLBACK_STATES:
            FALLBACK_STATES[county_name].classify_and_report(totals)
            updated.append(county_name)
        elif county_name in live_feed.LIVE_STATES:
            live_feed.LIVE_STATES[county_name].classify_and_report(totals)
            updated.append(county_name)

    return updated


if __name__ == "__main__":
    data = fetch_race()
    diagnose_structure(data)
    print()
    breakdown = find_county_breakdown(data)
    if breakdown:
        print("County breakdown found:")
        print(breakdown)
    else:
        print("No county breakdown found — statewide only:")
        print(get_statewide_totals(data))
