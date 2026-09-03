"""The whole-sequence leakage-safe split assignment for IO-VNBD synchronised.

Why these assignments (PRD §6.2 + the two Day-1 findings in
ml/docs/IO-VNBD-verification.md):

  * The independent unit is the *drive*, not the CSV clip. Driver E's Vta/Vtb/Vw
    families are each ONE continuous road trip chopped into consecutive clips
    (vta02, vta03, ... progress north along one road). Splitting clip-wise
    leaks road, weather, tyre state, mount angle across train/test.
    => split by ROUTE FAMILY.

  * IO-VNBD is one vehicle; the synchronised drivers are ~80 % Driver E; the
    phone mount is inconsistent (Driver D / Y1 is unusable). So:
      - headline "unseen route" test = a whole Driver-E route family (Vta, Peak
        District) whose roads never appear in train (Vw = Worcestershire/west,
        M = Coventry). Driver + vehicle ARE shared with train — the honest limit
        of what IO-VNBD supports, stated in ml/splits/README.md.
      - "unseen driver" test = ALL of Driver A, never trained.
      - Driver B (one drive, M) goes in train to add a second driver there.
      - Driver D (Y1) is excluded entirely — its phone stream is decoupled
        from the vehicle (sync_speed_corr 0.08).

  * Repeat-route pair for Kamal's magnetic route memory (FR-30): IO-VNBD's
    synchronised set has no clean "same road, driven twice" pair. The closest
    is a shared motorway corridor between vfa02 and vtb05 (~70 km of ~500 m
    grid cells in common, both Driver E). Both are held out of train so the
    velocity model never sees that geometry. True route repeats, if any, live
    in the unsynchronised Vw/Vta families (not pulled).
"""
from __future__ import annotations

# split name -> set of route families or explicit seq_ids assigned to it.
# Applied in order; an explicit seq_id override wins over its family.
SPLIT_BY_FAMILY: dict[str, set[str]] = {
    "train":            {"Vw", "M"},     # Vw = biggest E route pool (incl. the 214 km vw04)
    "val":              {"Vtb"},
    "test_id":          {"Vta"},         # headline: unseen route, seen driver.
                                         # Vta = Peak District, hilly — a hard test,
                                         # and matches the hill-corridor persona/map.
    "test_ood_driver":  {"S"},           # unseen driver (Driver A)
}

SPLIT_BY_SEQ_ID: dict[str, str] = {
    # repeat-route corridor pair for FR-30 — pulled OUT of val/train
    "vfa02": "test_repeat_corridor",
    "vtb05": "test_repeat_corridor",
    # Vf family: vfa01 is tiny and overlaps the Vta start area — exclude to keep
    # train/test geography clean (vfa02 is handled above).
    "vfa01": "excluded",
    # vehicle never moves in these (move_fraction 0.0) — no label signal
    "vw01":  "excluded",
    "vw15":  "excluded",
    # Driver D: phone decoupled from vehicle, unusable
    "y1":    "excluded",
}

ALL_SPLITS = ["train", "val", "test_id", "test_ood_driver", "test_repeat_corridor", "excluded"]


def assign_split(seq_id: str, route_family: str) -> str:
    if seq_id in SPLIT_BY_SEQ_ID:
        return SPLIT_BY_SEQ_ID[seq_id]
    for split, fams in SPLIT_BY_FAMILY.items():
        if route_family in fams:
            return split
    return "excluded"


def assign_all(sequences) -> dict[str, list]:
    out: dict[str, list] = {s: [] for s in ALL_SPLITS}
    for seq in sequences:
        out[assign_split(seq.seq_id, seq.route_family)].append(seq)
    return out
