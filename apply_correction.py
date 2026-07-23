"""
ONE-TIME retroactive correction.
Reconstructs the real Early vs Day-Of split for every non-Maricopa county
using the actual batch sequence, then re-publishes the corrected snapshot
and classifier state to the Gist.

Run this ONCE, locally, with GIST_ID and GIST_TOKEN set as environment
variables (same ones Render uses).
"""
import os
import az_bayesian_model as model
import az_publish as publish

GIST_ID = os.environ["GIST_ID"]
GIST_TOKEN = os.environ["GIST_TOKEN"]

# Restore whatever state currently exists first (preserves Maricopa's real data, etc.)
publish.restore_state_from_gist(GIST_ID, GIST_TOKEN)

CORRECTIONS = {
    "Greenlee": (142, 16, 31, 143, 22, 34),
    "Santa Cruz": (668, 113, 194, 375, 59, 66),
    "Yuma": (5746, 882, 1116, 1331, 190, 234),
    "Graham": (2304, 363, 349, 193, 21, 26),
    "Coconino": (3394, 671, 747, 1324, 203, 265),
    "Cochise": (5865, 1177, 1329, 2614, 320, 307),
    "Pima": (33354, 6202, 7292, 10972, 1298, 1064),
    "Yavapai": (22488, 3404, 3339, 5809, 531, 502),
    "Apache": (1667, 234, 350, 824, 141, 345),
    "Pinal": (18532, 2786, 2719, 7426, 742, 696),
    "Mohave": (17719, 2484, 2627, 4499, 637, 564),
    "Navajo": (3605, 590, 645, 2567, 299, 416),
    "Gila": (4238, 689, 686, 1084, 119, 122),
    "La Paz": (861, 108, 234, 333, 41, 88),
}

for name, (eb, es, eo, db, ds, do) in CORRECTIONS.items():
    county = model.COUNTIES[name]
    county.report("early", eb, es, eo)
    county.report("dayof", db, ds, do)

    # Sync the classifier so future updates correctly accumulate from here
    clf = model.FIRST_UPDATE_CLASSIFIERS[name]
    clf.first_update_done = True
    clf.cumulative_recorded = (eb+db, es+ds, eo+do)

    print(f"{name}: early={eb},{es},{eo}  dayof={db},{ds},{do}")

snap = publish.publish_snapshot(GIST_ID, GIST_TOKEN)
print()
print(f"Published corrected snapshot. {snap['statewide']['pctIn']:.1%} of vote in, "
      f"P(Biggs)={snap['statewide']['pBiggs']:.1f}%")
