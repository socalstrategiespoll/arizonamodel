import time
import xml.etree.ElementTree as ET
import requests

import az_bayesian_model as model

SOS_SUMMARY_URL = "https://apps.azsos.gov/ElectionResults/2026/State/2026%20Primary%20Election/Results.Summary.xml"
MIN_POLL_INTERVAL_SECONDS = 120

CONTEST_NAME_HINTS = ["governor", "gov"]
PARTY_HINT = "rep"

TRUST_LOW = 0.55
TRUST_HIGH = 0.67

COUNTY_NAME_MAP = {
    "MARICOPA": "Maricopa", "PIMA": "Pima", "YAVAPAI": "Yavapai", "PINAL": "Pinal",
    "MOHAVE": "Mohave", "COCHISE": "Cochise", "YUMA": "Yuma", "NAVAJO": "Navajo",
    "COCONINO": "Coconino", "GILA": "Gila", "APACHE": "Apache", "GRAHAM": "Graham",
    "LA PAZ": "La Paz", "LAPAZ": "La Paz", "SANTA CRUZ": "Santa Cruz",
    "SANTACRUZ": "Santa Cruz", "GREENLEE": "Greenlee",
}

CANDIDATE_NAME_MAP = {
    "B": None,
    "S": None,
    "O": None,
}


class FeedParseError(Exception):
    pass


def fetch_raw(url=SOS_SUMMARY_URL, timeout=30):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def diagnose_structure(xml_text, max_tags=40):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print("XML failed to parse:", e)
        print(xml_text[:1000])
        return

    tags_seen = set()

    def walk(el, depth=0):
        tags_seen.add(el.tag)
        if depth < 3:
            for child in el:
                walk(child, depth + 1)

    walk(root)
    print("Root tag:", root.tag)
    print("Distinct tags found (top levels):")
    for t in sorted(tags_seen)[:max_tags]:
        print(" -", t)
    print()
    print("First-level children of root:")
    for child in list(root)[:10]:
        print(" -", child.tag, child.attrib)


def parse_county_totals(xml_text):
    root = ET.fromstring(xml_text)

    contest_el = None
    for contest in root.iter():
        tag_lower = contest.tag.lower()
        if "contest" in tag_lower:
            name_attr = (contest.get("name") or contest.get("Name") or
                         contest.get("title") or contest.get("Title") or "")
            if any(h in name_attr.lower() for h in CONTEST_NAME_HINTS) and \
               PARTY_HINT in name_attr.lower():
                contest_el = contest
                break

    if contest_el is None:
        raise FeedParseError(
            "Could not locate the target contest in the feed. Call "
            "diagnose_structure() on this XML to see actual tag/attribute names, "
            "then update CONTEST_NAME_HINTS / PARTY_HINT or the matching logic below."
        )

    county_totals = {}
    for choice in contest_el.iter():
        tag_lower = choice.tag.lower()
        if "choice" not in tag_lower and "candidate" not in tag_lower:
            continue
        cand_name = (choice.get("name") or choice.get("Name") or
                     choice.get("candidate") or choice.get("Candidate") or "")
        cand_key = CANDIDATE_NAME_MAP_LOOKUP(cand_name)
        if cand_key is None:
            continue

        for county_el in choice.iter():
            ctag = county_el.tag.lower()
            if "county" not in ctag:
                continue
            county_name_raw = (county_el.get("name") or county_el.get("Name") or "").strip().upper()
            county_name = COUNTY_NAME_MAP.get(county_name_raw)
            if county_name is None:
                continue
            votes_str = (county_el.get("votes") or county_el.get("Votes") or
                         county_el.text or "0")
            try:
                votes = int(votes_str.strip())
            except (ValueError, AttributeError):
                continue
            county_totals.setdefault(county_name, {"B": 0, "S": 0, "O": 0})
            county_totals[county_name][cand_key] = votes

    if not county_totals:
        raise FeedParseError(
            "Contest found but no county-level vote counts were extracted. "
            "Call diagnose_structure() and inspect the contest_el subtree directly."
        )

    return county_totals


def CANDIDATE_NAME_MAP_LOOKUP(cand_name):
    cand_name_lower = cand_name.lower()
    for key, matcher in CANDIDATE_NAME_MAP.items():
        if matcher is not None and matcher.lower() in cand_name_lower:
            return key
    return None


class LiveCountyState:
    def __init__(self, name):
        self.name = name
        self.mode = None
        self.early_locked_totals = None
        self.modeled_early_threshold = None

    def classify_and_report(self, cumulative_bso):
        b, s, o = cumulative_bso["B"], cumulative_bso["S"], cumulative_bso["O"]
        cumulative_n = b + s + o
        county = model.COUNTIES[self.name]
        projected_total = county.total

        if self.mode is None:
            if cumulative_n == 0:
                return
            dump_pct = cumulative_n / projected_total
            if TRUST_LOW <= dump_pct <= TRUST_HIGH:
                self.mode = "normal"
                self.early_locked_totals = (b, s, o)
                county.report("early", b, s, o)
            elif dump_pct > TRUST_HIGH:
                self.mode = "high_dump"
                self.early_locked_totals = (b, s, o)
                county.report("early", b, s, o)
                print(f"[{self.name}] HIGH DUMP: {dump_pct:.1%} of projected total "
                      f"in first report — check total-turnout projection.")
            else:
                self.mode = "low_dump_exception"
                self.modeled_early_threshold = projected_total * (
                    county.early_total_proj / county.total
                )
                county.report("early", b, s, o)
                print(f"[{self.name}] LOW DUMP: {dump_pct:.1%} of projected total — "
                      f"treating as incomplete Wave 1 (threshold "
                      f"{self.modeled_early_threshold:,.0f}).")
                if cumulative_n >= self.modeled_early_threshold:
                    self.mode = "normal"
                    self.early_locked_totals = (b, s, o)
            return

        if self.mode == "low_dump_exception":
            county.report("early", b, s, o)
            if cumulative_n >= self.modeled_early_threshold:
                self.mode = "normal"
                self.early_locked_totals = (b, s, o)
                print(f"[{self.name}] Crossed modeled Early threshold — now Day-Of mode.")
            return

        eb, es, eo = self.early_locked_totals
        dayof_b = max(0, b - eb)
        dayof_s = max(0, s - es)
        dayof_o = max(0, o - eo)
        county.report("early", eb, es, eo)
        county.report("dayof", dayof_b, dayof_s, dayof_o)


LIVE_STATES = {name: LiveCountyState(name) for name in model.COUNTIES}


def poll_once(url=SOS_SUMMARY_URL):
    xml_text = fetch_raw(url)
    county_totals = parse_county_totals(xml_text)
    for county_name, bso in county_totals.items():
        LIVE_STATES[county_name].classify_and_report(bso)
    return county_totals


def run_polling_loop(url=SOS_SUMMARY_URL, interval=MIN_POLL_INTERVAL_SECONDS):
    interval = max(interval, MIN_POLL_INTERVAL_SECONDS)
    while True:
        try:
            poll_once(url)
            model.report_status()
        except FeedParseError as e:
            print("Parse error:", e)
        except requests.RequestException as e:
            print("Fetch error:", e)
        time.sleep(interval)


if __name__ == "__main__":
    xml_text = fetch_raw()
    diagnose_structure(xml_text)
