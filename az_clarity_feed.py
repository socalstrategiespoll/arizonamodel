import io
import re
import zipfile
import xml.etree.ElementTree as ET

import requests

import az_bayesian_model as model

CLARITY_COUNTIES = {
    "Pinal": "Pinal",
    "Yuma": "Yuma",
    "Greenlee": "Greenlee",
}

CONTEST_MATCH = re.compile(r"\bgovernor\b", re.IGNORECASE)
PARTY_MATCH = re.compile(r"\brep\b", re.IGNORECASE)

CANDIDATE_TO_KEY = {
    "BIGGS": "B",
    "SCHWEIKERT": "S",
}

VOTE_TYPE_TO_BUCKET_RULES = [
    (re.compile(r"election\s*day", re.IGNORECASE), "dayof"),
    (re.compile(r"provisional", re.IGNORECASE), "dayof"),
    (re.compile(r"absentee|mail|early", re.IGNORECASE), "early"),
]


class ClarityFeedError(Exception):
    pass


def county_base_url(county):
    return f"https://results.enr.clarityelections.com/AZ/{county}/"


def find_current_election_path(county, timeout=30):
    url = county_base_url(county)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    matches = re.findall(r'href="(\d+)/?"', html)
    if not matches:
        matches = re.findall(r"/(\d{5,7})/", html)
    if not matches:
        raise ClarityFeedError(
            f"Could not find any election ID folder on {url}. "
            "Inspect the page HTML directly to find the current structure."
        )

    election_id = sorted(set(matches), key=int)[-1]
    return f"{url}{election_id}/"


def fetch_current_version(election_path, timeout=30):
    url = election_path + "current_ver.txt"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text.strip()


def fetch_detail_xml(election_path, version, timeout=60):
    candidates = [
        f"{election_path}{version}/reports/detail.xml",
        f"{election_path}{version}/reports/detailxml.zip",
    ]
    last_error = None
    for url in candidates:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            if url.endswith(".zip"):
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
                if not xml_names:
                    raise ClarityFeedError(f"No XML file found inside {url}")
                return zf.read(xml_names[0]).decode("utf-8", errors="replace")
            return resp.text
        except (requests.RequestException, zipfile.BadZipFile) as e:
            last_error = e
            continue
    raise ClarityFeedError(
        f"Could not fetch detail report from any known path. Last error: {last_error}. "
        f"Tried: {candidates}"
    )


def diagnose_structure(xml_text, max_items=30):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print("XML failed to parse:", e)
        print(xml_text[:1000])
        return

    print("Root tag:", root.tag)
    contests = root.findall(".//Contest")
    print(f"Found {len(contests)} <Contest> elements")
    for c in contests[:max_items]:
        print(" -", c.get("text") or c.get("name") or c.attrib)

    vote_types = set()
    for vt in root.findall(".//VoteType"):
        name = vt.get("name") or vt.get("text")
        if name:
            vote_types.add(name)
    print("Distinct VoteType names found:", vote_types)


def classify_vote_type(name):
    for pattern, bucket in VOTE_TYPE_TO_BUCKET_RULES:
        if pattern.search(name):
            return bucket
    return None


def parse_county_totals(xml_text):
    root = ET.fromstring(xml_text)

    contest_el = None
    for contest in root.findall(".//Contest"):
        name_attr = contest.get("text") or contest.get("name") or ""
        if CONTEST_MATCH.search(name_attr) and (
            PARTY_MATCH.search(name_attr) or "rep" in (contest.get("party", "").lower())
        ):
            contest_el = contest
            break

    if contest_el is None:
        raise ClarityFeedError(
            "Could not locate the target contest in the detail XML. "
            "Call diagnose_structure() on this XML to see actual contest names."
        )

    totals = {"B": {"early": 0, "dayof": 0}, "S": {"early": 0, "dayof": 0}, "O": {"early": 0, "dayof": 0}}

    for choice in contest_el.findall("Choice"):
        cand_name = (choice.get("text") or choice.get("name") or "").upper()
        cand_key = "O"
        for surname, key in CANDIDATE_TO_KEY.items():
            if surname in cand_name:
                cand_key = key
                break

        for vt in choice.findall("VoteType"):
            vt_name = vt.get("name") or vt.get("text") or ""
            bucket = classify_vote_type(vt_name)
            if bucket is None:
                continue
            votes_str = vt.get("votes") or "0"
            try:
                votes = int(votes_str)
            except ValueError:
                continue
            totals[cand_key][bucket] += votes

    return totals


def update_model_from_clarity(county, xml_text=None):
    if xml_text is None:
        election_path = find_current_election_path(county)
        version = fetch_current_version(election_path)
        xml_text = fetch_detail_xml(election_path, version)

    totals = parse_county_totals(xml_text)

    model_county = model.COUNTIES[county]
    model_county.report("early", totals["B"]["early"], totals["S"]["early"], totals["O"]["early"])
    model_county.report("dayof", totals["B"]["dayof"], totals["S"]["dayof"], totals["O"]["dayof"])
    return totals


def update_all_clarity_counties():
    results = {}
    for county in CLARITY_COUNTIES:
        try:
            results[county] = update_model_from_clarity(county)
        except (ClarityFeedError, requests.RequestException) as e:
            print(f"[{county}] failed: {e}")
    return results


if __name__ == "__main__":
    for county in CLARITY_COUNTIES:
        print(f"=== {county} ===")
        try:
            path = find_current_election_path(county)
            print("Election path:", path)
            version = fetch_current_version(path)
            print("Current version:", version)
            xml_text = fetch_detail_xml(path, version)
            diagnose_structure(xml_text)
            print(parse_county_totals(xml_text))
        except (ClarityFeedError, requests.RequestException) as e:
            print(f"Failed for {county}: {e}")
        print()
