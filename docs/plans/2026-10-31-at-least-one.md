# Design: `at_least_one` -- an answer where the thresholds give none

## Goal
An optional request setting: when no label of a multilabel model passes its tuned threshold,
return the best label anyway, marked `above_threshold: false`. For an editor who knows a text
belongs somewhere in the label set and would rather see the nearest subject than nothing.

## Why not what exists
- `top_k=1` gives one label always -- and cuts the real second label from every answer that
  had two.
- `threshold` lowers the cut for every label of every text, so it changes answers that were
  fine.
`at_least_one` touches only the answers that are empty, and leaves every other one as it was.

## Measured (2026-10-31)
A multilabel subject model (`ccm:taxonid`, 45 labels, profile auto) trained on 80 % of
`data_30k.csv`, measured on the other 20 % -- 6,504 rows it never saw (seed 42):

| Variant | Precision | Recall | Micro F1 | Per-text F1 | Empty | Added | Added correct |
|---|---|---|---|---|---|---|---|
| today (empty stays empty) | 0.872 | 0.897 | 0.885 | 0.888 | 3.2 % | -- | -- |
| **at least one** | 0.865 | 0.908 | 0.886 | 0.903 | 0 % | 206 | 51 % |
| ... floor 75 % of the threshold | 0.871 | 0.901 | 0.886 | 0.893 | 2.3 % | 56 | 64 % |
| ... floor 50 % | 0.869 | 0.904 | 0.886 | 0.897 | 1.4 % | 113 | 58 % |
| ... floor 25 % | 0.867 | 0.907 | 0.886 | 0.901 | 0.5 % | 176 | 54 % |

- The cost is small because the option only acts on the 3.2 % of texts that are empty today:
  precision -0.7 points, recall +1.1, per-text F1 +1.5, micro F1 unchanged.
- An added label is right about half the time -- which is why it is marked and never passes
  as an answer the model is sure of.
- A floor (only if the best label reaches a share of its threshold) buys a higher hit rate
  for more empty answers and no F1. **Not built**: one more lever for no measurable gain.
  The editor's own knowledge that the text belongs somewhere is the better filter.

## Behaviour
- Multilabel decision only. Ranking (`top_k`) already returns N labels; binary and multiclass
  decide by argmax and always answer.
- Applied after `label_filter` and after a request's `threshold`: the best among the requested
  labels; a label above the request's own cut is a normal answer.
- Off by default; reported back in `applied_settings.at_least_one`.
- `/predict`, `/predict/multi` (JSON field), `/predict/csv` (form field) and the admin UI's
  Query tab (a checkbox; the stand-in shows greyed with "below threshold", in the bulk table
  too, and the table's download gains an `above_threshold` column only when one occurs).
- Not in `POST /models/{name}/evaluate`: evaluation measures the model's own decisions.

## Files
`app/classifier.py` (the rule), `app/schemas/serving.py`, `app/routes/predict.py`,
`app/routes/predict_bulk.py`, `app/predict_csv.py`; UI `query.js`, `index.html`, strings.
Tests: `tests/test_at_least_one.py`, `tests/test_api.py`, UI tests.
