import csv
import io
import re

import requests

import az_bayesian_model as model

MARICOPA_RESULTS_PAGE = "https://elections.maricopa.gov/results-and-data/election-results.html"

CONTEST_MATCH = re.compile(r"\bgovernor\b", re.IGNORECASE)
PARTY_MATCH = re.compile(r"\brep\b", re.IGNORECASE)

CANDIDATE_TO_KEY = {
    "BIGGS": "B",
    "SCHWEIKERT": "S",
}

COUNTING_GROUP_TO_BUCKET = {
    "Early Vote": "early",
    "Election Day": "dayof",
    "Early A.R.S. 16-579": "dayof",
    "Provisional": "dayof",
}


class MaricopaFeedError(Exception):
    pass


def find_current_results_txt_url():
    resp = requests.get(MARICOPA_RESULTS_PAGE, timeout=30)
    resp.raise_for_status()
    html = resp.text

    matches = re.findall(r'href="([^"]+\.txt)"', html, re.IGNORECASE)
    if not matches:
        raise MaricopaFeedError(
            "Could not find any .txt link on the Maricopa results page. "
            "The page structure may have changed — inspect the HTML directly."
        )

    url = matches[0]
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://elections.maricopa.gov" + url
    elif not url.startswith("http"):
        url = "https://elections.maricopa.gov/" + url.lstrip("/")
    return url


def fetch_results_txt(url=None):
    if url is None:
        url = find_current_results_txt_url()
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def sniff_dialect(sample_text):
    sample = sample_text[:5000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|")
        return dialect
    except csv.Error:
        class FallbackDialect(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\r\n"
            quoting = csv.QUOTE_MINIMAL
        return FallbackDialect


def parse_maricopa_governor_totals(results_text):
    dialect = sniff_dialect(results_text)
    reader = csv.DictReader(io.StringIO(results_text), dialect=dialect)

    if reader.fieldnames is None:
        raise MaricopaFeedError("File has no header row — cannot map columns.")

    fieldnames_lower = {f.lower(): f for f in reader.fieldnames}

    def col(name):
        return fieldnames_lower.get(name.lower())

    contest_col = col("ContestName")
    contest_party_col = col("ContestPartyAffiliation")
    candidate_col = col("CandidateName")

    if not all([contest_col, contest_party_col, candidate_col]):
        raise MaricopaFeedError(
            f"Expected columns not found. Actual header: {reader.fieldnames}"
        )

    vote_cols = {}
    for group_name in COUNTING_GROUP_TO_BUCKET:
        c = col(f"Votes_{group_name}")
        if c:
            vote_cols[group_name] = c

    if not vote_cols:
        raise MaricopaFeedError(
            f"No Votes_<CountingGroup> columns found. Actual header: {reader.fieldnames}"
        )

    totals = {"B": {"early": 0, "dayof": 0}, "S": {"early": 0, "dayof": 0}, "O": {"early": 0, "dayof": 0}}

    for row in reader:
        contest_name = row.get(contest_col, "")
        party = row.get(contest_party_col, "")
        if not CONTEST_MATCH.search(contest_name):
            continue
        if not PARTY_MATCH.search(party) and not PARTY_MATCH.search(contest_name):
            continue

        cand_raw = row.get(candidate_col, "").upper()
        cand_key = "O"
        for surname, key in CANDIDATE_TO_KEY.items():
            if surname in cand_raw:
                cand_key = key
                break

        for group_name, bucket in COUNTING_GROUP_TO_BUCKET.items():
            vcol = vote_cols.get(group_name)
            if not vcol:
                continue
            raw_val = row.get(vcol, "0").strip()
            try:
                votes = int(raw_val) if raw_val else 0
            except ValueError:
                continue
            totals[cand_key][bucket] += votes

    return totals


def update_model_from_maricopa(results_text=None, url=None):
    if results_text is None:
        results_text = fetch_results_txt(url)
    totals = parse_maricopa_governor_totals(results_text)

    county = model.COUNTIES["Maricopa"]
    county.report("early", totals["B"]["early"], totals["S"]["early"], totals["O"]["early"])
    county.report("dayof", totals["B"]["dayof"], totals["S"]["dayof"], totals["O"]["dayof"])
    return totals


if __name__ == "__main__":
    url = find_current_results_txt_url()
    print("Resolved results file URL:", url)
    text = fetch_results_txt(url)
    print("First 500 chars:")
    print(text[:500])
    totals = parse_maricopa_governor_totals(text)
    print(totals)
