# Training set audit — Section 1.5 reference

**Real positive crops:** 14
**Real negative crops:** 22
**Synthetic positive crops:** 51

## Real positive training sites (source of `data/crops/positive/`)

- `de_bad_lauchstaedt` — 3 crop(s)
- `de_weisweiler` — 1 crop(s)
- `nl_bergermeer` — 1 crop(s)
- `nl_grijpskerk` — 1 crop(s)
- `ro_totea` — 3 crop(s)
- `silesia_knurow` — 1 crop(s)
- `silesia_pniowek` — 1 crop(s)
- `silesia_rybnik` — 1 crop(s)
- `silesia_zofiowka` — 2 crop(s)

## Real negative training sites (source of `data/crops/negative/`)

- `belchatow` — 1 crop(s)
- `ctrl_dk_jutland` — 1 crop(s)
- `ctrl_nl_flevoland` — 1 crop(s)
- `ctrl_ro_wallachia` — 1 crop(s)
- `dunkerque` — 1 crop(s)
- `groningen_gas_field` — 2 crop(s)
- `groningen_polder_E` — 2 crop(s)
- `groningen_polder_N` — 2 crop(s)
- `groningen_polder_NW` — 1 crop(s)
- `groningen_polder_SE` — 1 crop(s)
- `groningen_polder_SW` — 2 crop(s)
- `janschwalde` — 1 crop(s)
- `larobla` — 1 crop(s)
- `maritsa_east2` — 1 crop(s)
- `neurath` — 1 crop(s)
- `philippsburg` — 1 crop(s)
- `rovinari` — 1 crop(s)
- `tusimice` — 1 crop(s)

## Synthetic plume substrates (which real terrain was used to generate synthetics)

- `belchatow` — 3 synthetic plume(s)
- `ctrl_dk_jutland` — 3 synthetic plume(s)
- `ctrl_nl_flevoland` — 3 synthetic plume(s)
- `ctrl_ro_wallachia` — 3 synthetic plume(s)
- `dunkerque` — 3 synthetic plume(s)
- `groningen_gas_field` — 6 synthetic plume(s)
- `groningen_polder_E` — 6 synthetic plume(s)
- `groningen_polder_N` — 6 synthetic plume(s)
- `groningen_polder_NW` — 3 synthetic plume(s)
- `groningen_polder_SW` — 3 synthetic plume(s)
- `larobla` — 3 synthetic plume(s)
- `maritsa_east2` — 3 synthetic plume(s)
- `rovinari` — 3 synthetic plume(s)
- `tusimice` — 3 synthetic plume(s)

## Candidate site classification for evaluation defensibility

| Site | Training status | Treat as test result if model... |
|---|---|---|
| belchatow | `training_negative_and_synthetic_substrate` | **In training as negative AND used as synthetic substrate** — model was told negative, but synthetic plumes were generated on this terrain. Positive detection at test time is suggestive but partially leaked. |
| rybnik | `training_positive` | **In training as positive** — model was told this site is methane. Cannot be a held-out test; performance here is in-sample. |
| weisweiler | `training_positive` | **In training as positive** — model was told this site is methane. Cannot be a held-out test; performance here is in-sample. |
| lippendorf | `held_out` | **TRULY held-out** — model never saw this site's tiles in any form during training. Performance here is a clean independent test. |
| neurath | `training_negative` | **In training as negative** — model was told this site is NOT methane. A positive detection at test time is stronger evidence than pure held-out, because the model is overriding its own training label. |
| boxberg | `held_out` | **TRULY held-out** — model never saw this site's tiles in any form during training. Performance here is a clean independent test. |
| groningen | `training_negative_and_synthetic_substrate` | **In training as negative AND used as synthetic substrate** — model was told negative, but synthetic plumes were generated on this terrain. Positive detection at test time is suggestive but partially leaked. |
| maasvlakte | `held_out` | **TRULY held-out** — model never saw this site's tiles in any form during training. Performance here is a clean independent test. |

## Section 1.5 — proposed text

The European fine-tuning dataset contains 14 real positive crops from 9 sites, 51 synthetic positive crops generated from 14 negative background tiles, and 22 real negative crops from 18 sites.

**Truly held-out candidate sites** (never seen during training, valid as an independent test set): lippendorf, boxberg, maasvlakte.

**Candidate sites in training as negatives** (positive detection at test time = model overrides its training label, stronger than mere held-out): belchatow, neurath, groningen.

**Candidate sites in training as positives** (in-sample, not valid as independent test): rybnik, weisweiler.
