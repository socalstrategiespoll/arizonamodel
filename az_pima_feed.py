import csv
import io
import re

import requests

import az_bayesian_model as model

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


PIMA_RESULTS_PAGE = "https://www.pima.gov/2865/Election-Results"

CONTEST_NAME_TARGET = "governor"
PARTY_TARGET = "REP"

CANDIDATE_TO_KEY = {
    "BIGGS": "B",
    "SCHWEIKERT": "S",
}

GROUP_TO_BUCKET = {
    "Election Day": "dayof",
    "Provisional": "dayof",
    "Early": "early",
}

NON_CANDIDATE_LABELS = {"WRITE-IN", "OVER VOTES", "UNDER VOTES"}


class PimaFeedError(Exception):
    pass


def find_current_csv_url(timeout=12):
    resp = requests.get(PIMA_RESULTS_PAGE, timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    html = resp.text

    pattern = re.compile(
        r'href="([^"]+)"[^>]*>\s*\[?\s*(?:2026\s+)?Primary Election[^<]*Group Detail[^<]*(?:Excel|CSV)[^<]*',
        re.IGNORECASE
    )
    match = pattern.search(html)
    if not match:
        pattern2 = re.compile(r'href="([^"]+)"[^>]*>[^<]*Group Detail[^<]*', re.IGNORECASE)
        match = pattern2.search(html)

    if not match:
        raise PimaFeedError(
            "Could not find a 'Group Detail' results link on the Pima results page "
            "(it likely hasn't been posted for this election yet, or the link text "
            "changed). Inspect the page HTML directly to find the current link."
        )

    url = match.group(1)
    if url.startswith("/"):
        url = "https://www.pima.gov" + url
    return url


def fetch_csv_text(url, timeout=15):
    resp = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    return resp.text


def parse_header_rows(lines):
    row1 = next(csv.reader([lines[0]]))
    row2 = next(csv.reader([lines[1]]))
    row3 = next(csv.reader([lines[2]]))

    n = len(row1)
    columns = []
    for i in range(n):
        contest = row1[i] if i < len(row1) else ""
        party = row2[i] if i < len(row2) else ""
        candidate = row3[i] if i < len(row3) else ""
        columns.append((contest.strip(), party.strip(), candidate.strip()))
    return columns


def find_target_columns(columns, contest_target=CONTEST_NAME_TARGET, party_target=PARTY_TARGET):
    matches = []
    for i, (contest, party, candidate) in enumerate(columns):
        if contest_target.lower() in contest.lower() and party.upper() == party_target.upper():
            matches.append(i)
    return matches


def diagnose_structure(csv_text, max_contests=40):
    lines = csv_text.splitlines()
    columns = parse_header_rows(lines)

    contests_seen = set()
    for contest, party, candidate in columns:
        if contest:
            contests_seen.add((contest, party))

    print(f"Found {len(contests_seen)} distinct (contest, party) pairs")
    for c in sorted(contests_seen)[:max_contests]:
        print(" -", c)

    groups_seen = set()
    for line in lines[3:20]:
        row = next(csv.reader([line]))
        if len(row) > 1:
            groups_seen.add(row[1])
    print("Group values seen in first data rows:", groups_seen)


def parse_county_totals(csv_text, contest_target=CONTEST_NAME_TARGET, party_target=PARTY_TARGET):
    lines = csv_text.splitlines()
    if len(lines) < 4:
        raise PimaFeedError("File has fewer than 4 lines — cannot contain header + data.")

    columns = parse_header_rows(lines)
    target_col_indices = find_target_columns(columns, contest_target, party_target)

    if not target_col_indices:
        raise PimaFeedError(
            f"Could not find contest matching '{contest_target}' / party '{party_target}'. "
            "Call diagnose_structure() to see available (contest, party) pairs."
        )

    totals = {"B": {"early": 0, "dayof": 0}, "S": {"early": 0, "dayof": 0}, "O": {"early": 0, "dayof": 0}}

    reader = csv.reader(lines[3:])
    for row in reader:
        if len(row) < 2:
            continue
        group = row[1].strip()
        bucket = GROUP_TO_BUCKET.get(group)
        if bucket is None:
            continue

        for col_idx in target_col_indices:
            if col_idx >= len(row):
                continue
            candidate_label = columns[col_idx][2].upper()
            if candidate_label in NON_CANDIDATE_LABELS:
                cand_key = "O" if candidate_label == "WRITE-IN" else None
            else:
                cand_key = "O"
                for surname, key in CANDIDATE_TO_KEY.items():
                    if surname in candidate_label:
                        cand_key = key
                        break

            if cand_key is None:
                continue

            val_str = row[col_idx].strip()
            try:
                votes = int(val_str) if val_str else 0
            except ValueError:
                continue
            totals[cand_key][bucket] += votes

    return totals


def update_model_from_pima(csv_url=None, csv_text=None):
    if csv_text is None:
        if csv_url is None:
            csv_url = find_current_csv_url()
        csv_text = fetch_csv_text(csv_url)
    totals = parse_county_totals(csv_text)

    county = model.COUNTIES["Pima"]
    county.report("early", totals["B"]["early"], totals["S"]["early"], totals["O"]["early"])
    county.report("dayof", totals["B"]["dayof"], totals["S"]["dayof"], totals["O"]["dayof"])
    return totals


if __name__ == "__main__":
    test_url = "https://www.pima.gov/asset/7502018f-6bc2-4425-801c-75949d8b3b81"
    csv_text = fetch_csv_text(test_url)
    diagnose_structure(csv_text)
