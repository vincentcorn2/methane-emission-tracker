"""
scripts/document_training_data.py
===================================
Audit the actual training data composition and emit a report-ready paragraph
for Section 1.5 of the EIB report.

Reads:
  data/crops/positive/*_label.json
  data/crops/negative/*_label.json
  data/crops/synthetic/*_label.json
  data/crops/manifest.json (if present)

Writes:
  results_analysis/training_set_audit.md   — markdown summary
  results_analysis/training_set_audit.json — machine-readable

Why this exists
---------------
The current Section 1.5 says "14 real positives + 51 synthetic + 22 negatives"
without naming sites. A reviewer will ask which sites. This script answers:
  - which sites' tiles are in training as positives
  - which sites' tiles are in training as negatives
  - which candidate sites are TRULY held out (never seen during training)
  - which candidate sites are in training as negatives but later evaluated as
    positives (a stronger defensibility story than pure held-out)
"""
import json
from collections import defaultdict
from pathlib import Path

CROPS_DIR = Path("data/crops")
OUT_DIR = Path("results_analysis")

CANDIDATE_SITES = [
    "belchatow", "rybnik", "weisweiler", "lippendorf",
    "neurath", "boxberg", "groningen", "maasvlakte",
]


def slug_from_filename(stem: str) -> str:
    """Return a coarse site slug for grouping crops by origin."""
    # Positives:  silesia_rybnik_T34UCA_20240628_enh19  → silesia_rybnik
    # Negatives:  belchatow_T34UCB_20240824             → belchatow
    # Synthetics: synth_belchatow_T34UCB_20240824_000   → synth_belchatow
    parts = stem.split("_T")[0]
    return parts


def candidate_match(slug: str) -> str | None:
    """Map a training slug onto one of the eight candidate sites, if any."""
    s = slug.lower()
    for c in CANDIDATE_SITES:
        if c in s:
            return c
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pos = sorted((CROPS_DIR / "positive").glob("*.npy"))
    neg = sorted((CROPS_DIR / "negative").glob("*.npy"))
    syn = sorted((CROPS_DIR / "synthetic").glob("*.npy"))

    pos_by_site = defaultdict(list)
    neg_by_site = defaultdict(list)
    syn_by_source = defaultdict(int)

    for p in pos:
        if "_label" in p.stem:
            continue
        slug = slug_from_filename(p.stem)
        pos_by_site[slug].append(p.stem)

    for p in neg:
        if "_label" in p.stem:
            continue
        slug = slug_from_filename(p.stem)
        neg_by_site[slug].append(p.stem)

    for p in syn:
        if "_label" in p.stem:
            continue
        # Synthetic file name: synth_<background-site>_<tile>_<date>_<idx>.npy
        src = p.stem.split("_T")[0].replace("synth_", "")
        syn_by_source[src] += 1

    # Classify each candidate site
    classification = {}
    for c in CANDIDATE_SITES:
        in_pos = any(c in s.lower() for s in pos_by_site)
        in_neg = any(c in s.lower() for s in neg_by_site)
        in_syn = any(c in s.lower() for s in syn_by_source)

        if in_pos:
            cat = "training_positive"
        elif in_neg and in_syn:
            cat = "training_negative_and_synthetic_substrate"
        elif in_neg:
            cat = "training_negative"
        elif in_syn:
            cat = "synthetic_substrate_only"
        else:
            cat = "held_out"
        classification[c] = cat

    # Build audit object
    audit = {
        "counts": {
            "real_positive_crops": len([p for p in pos if "_label" not in p.stem]),
            "real_negative_crops": len([p for p in neg if "_label" not in p.stem]),
            "synthetic_positive_crops": len([p for p in syn if "_label" not in p.stem]),
        },
        "positive_sites": dict(pos_by_site),
        "negative_sites": dict(neg_by_site),
        "synthetic_sources": dict(syn_by_source),
        "candidate_classification": classification,
    }

    (OUT_DIR / "training_set_audit.json").write_text(json.dumps(audit, indent=2))

    # Emit markdown
    md = []
    md.append("# Training set audit — Section 1.5 reference\n")
    md.append(f"**Real positive crops:** {audit['counts']['real_positive_crops']}")
    md.append(f"**Real negative crops:** {audit['counts']['real_negative_crops']}")
    md.append(f"**Synthetic positive crops:** {audit['counts']['synthetic_positive_crops']}")
    md.append("")
    md.append("## Real positive training sites (source of `data/crops/positive/`)\n")
    for site, crops in sorted(pos_by_site.items()):
        md.append(f"- `{site}` — {len(crops)} crop(s)")
    md.append("")
    md.append("## Real negative training sites (source of `data/crops/negative/`)\n")
    for site, crops in sorted(neg_by_site.items()):
        md.append(f"- `{site}` — {len(crops)} crop(s)")
    md.append("")
    md.append("## Synthetic plume substrates (which real terrain was used to generate synthetics)\n")
    for src, n in sorted(syn_by_source.items()):
        md.append(f"- `{src}` — {n} synthetic plume(s)")
    md.append("")
    md.append("## Candidate site classification for evaluation defensibility\n")
    md.append("| Site | Training status | Treat as test result if model... |")
    md.append("|---|---|---|")
    interpretations = {
        "training_positive":
            "**In training as positive** — model was told this site is methane. Cannot be a held-out test; performance here is in-sample.",
        "training_negative":
            "**In training as negative** — model was told this site is NOT methane. A positive detection at test time is stronger evidence than pure held-out, because the model is overriding its own training label.",
        "training_negative_and_synthetic_substrate":
            "**In training as negative AND used as synthetic substrate** — model was told negative, but synthetic plumes were generated on this terrain. Positive detection at test time is suggestive but partially leaked.",
        "synthetic_substrate_only":
            "**Synthetic substrate only** — terrain seen as background for synthetic plumes but not labeled either way directly.",
        "held_out":
            "**TRULY held-out** — model never saw this site's tiles in any form during training. Performance here is a clean independent test.",
    }
    for site, cat in classification.items():
        md.append(f"| {site} | `{cat}` | {interpretations[cat]} |")
    md.append("")
    md.append("## Section 1.5 — proposed text\n")
    md.append("The European fine-tuning dataset contains "
              f"{audit['counts']['real_positive_crops']} real positive crops from "
              f"{len(pos_by_site)} sites, "
              f"{audit['counts']['synthetic_positive_crops']} synthetic positive crops "
              f"generated from {len(syn_by_source)} negative background tiles, "
              f"and {audit['counts']['real_negative_crops']} real negative crops from "
              f"{len(neg_by_site)} sites.\n")

    held_out = [s for s, c in classification.items() if c == "held_out"]
    train_neg = [s for s, c in classification.items()
                 if c in ("training_negative", "training_negative_and_synthetic_substrate")]
    train_pos = [s for s, c in classification.items() if c == "training_positive"]

    if held_out:
        md.append(f"**Truly held-out candidate sites** (never seen during training, "
                  f"valid as an independent test set): {', '.join(held_out)}.\n")
    if train_neg:
        md.append(f"**Candidate sites in training as negatives** (positive detection at "
                  f"test time = model overrides its training label, stronger than mere "
                  f"held-out): {', '.join(train_neg)}.\n")
    if train_pos:
        md.append(f"**Candidate sites in training as positives** (in-sample, not valid "
                  f"as independent test): {', '.join(train_pos)}.\n")

    (OUT_DIR / "training_set_audit.md").write_text("\n".join(md))

    # Console summary
    print("\n" + "=" * 70)
    print("TRAINING SET AUDIT")
    print("=" * 70)
    print(f"Positive crops:  {audit['counts']['real_positive_crops']} "
          f"({len(pos_by_site)} sites)")
    print(f"Negative crops:  {audit['counts']['real_negative_crops']} "
          f"({len(neg_by_site)} sites)")
    print(f"Synthetic crops: {audit['counts']['synthetic_positive_crops']} "
          f"({len(syn_by_source)} substrates)")
    print()
    print(f"{'Candidate':<14} {'Classification':<48}")
    print("-" * 70)
    for site, cat in classification.items():
        print(f"{site:<14} {cat}")
    print()
    print(f"Wrote: {OUT_DIR}/training_set_audit.md")
    print(f"Wrote: {OUT_DIR}/training_set_audit.json")


if __name__ == "__main__":
    main()
