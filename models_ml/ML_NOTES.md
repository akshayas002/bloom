# Bloom ML model — leakage diagnosis and fix

## The bug

The previous `bloom_model.pkl` reported an MAE of **0.09 days** — it claimed
to predict the next period to within about two hours. That number was a
symptom of a bug, not a sign of a good model.

**Root cause: target leakage.** `train_model.py` included the *current*
cycle's own completed length (`cycle_length`) as an input feature, alongside
`days_since_last_period`. But `cycle_length` — the full length of the cycle
being predicted — isn't knowable until that cycle is actually over. The
model didn't learn a real pattern; it learned to compute

```
days_until ≈ cycle_length − days_since_last_period
```

which is just rearranging two of its own inputs. I confirmed this by
feeding the old model random feature vectors and comparing its output to
that exact formula:

```
cyc=30.4 since=18.4  naive=11.92  pred=11.89
cyc=23.6 since=3.9   naive=19.73  pred=20.06
cyc=27.8 since=4.4   naive=23.40  pred=23.72
```

Predictions tracked the naive arithmetic almost exactly — that's the
leak, quantified.

**It compounded into a second bug: train/serve skew.** In production,
`cycle_service.py` filled that same `cycle_length` slot with `user.avg_cycle`
(a legitimate, historical average) — not the actual completed length the
model was trained on. So the reported 0.09-day MAE was never something the
deployed app could have achieved even if the leak were harmless: the model
was validated on one kind of input and served a different one under the
same feature name.

## The fix

1. **Removed the leaking feature entirely.** The model now only ever sees
   information that would genuinely be available at prediction time:
   `age`, `days_since_last_period`, same-day `mood`/`flow`/`symptom`/
   `stress`/`sleep`/`exercise`, `BMI`, and — standing in for "expected
   cycle length" — `avg_previous` (a rolling historical average over the
   user's own last up to 6 completed cycles) and `cycle_variation`
   (max − min over that same window). No feature describes the outcome of
   the cycle being predicted.

2. **One shared feature contract.** `app/ml/feature_spec.py` is now the
   single place that defines the category encodings and feature order.
   Both `train_model.py` and `predictor.py` import from it, so the
   train/serve mismatch that let the leak hide in the first place can't
   recur silently.

3. **Grouped, honest validation.** Cycles from the same user are highly
   correlated (same baseline, same lifestyle habits), so a random row-level
   train/test split would leak information about a user across the split
   even without the `cycle_length` bug. Validation is now grouped by
   `user_id`:
   - 5-fold **GroupKFold** cross-validation on the training users only.
   - A final **held-out set of users the model has never seen in any form**
     (20% of users, entirely separate), used for the MAE that actually
     gets reported and shipped in the model bundle.

4. **PCOS risk scoring** (`score_pcos_risk`) also now takes the historical
   average cycle length instead of a dead/unused parameter that used to be
   passed in but never actually read.

## Honest results (synthetic data — see below)

| Approach                                             | MAE (days) |
|-------------------------------------------------------|:---------:|
| Predict-the-mean baseline                              | 7.88      |
| Naive heuristic (`avg_previous − days_since_last`)      | 5.01      |
| **Fixed model, grouped CV (train users)**               | **2.98**  |
| **Fixed model, held-out unseen users**                  | **2.87**  |

Fuller picture on the held-out users: **R² = 0.775**, RMSE = 4.61 days,
55% of predictions land within ±2 days and 85% within ±5 days. The model
now beats both trivial baselines by a real margin, on users it has never
seen in any form — and these are numbers you can defend in an interview,
unlike the old 0.09.

## A second, independent bug found while computing these metrics

While computing R²/RMSE I noticed `predictor.py` was clipping the model's
*output* (`days_until`) to `[PHYS_MIN, PHYS_MAX] = [14, 60]` — bounds that
describe a physiologically plausible **cycle length**, not "days remaining
until the next period." Those are different quantities: someone checking
in the day before their period starts has `days_until ≈ 1`, which is
completely normal, not out of range.

Checking the actual label distribution confirmed the impact: **2,808 of
6,030 rows (46.6%)** have a true `days_until` below 14 — i.e. this bug was
silently flooring almost half of all real predictions at "at least 14
days," which would visibly mislead anyone checking in during the back half
of their cycle. This bug predates the leakage fix; it was already present
in the original `predictor.py`. Fixed by adding `DAYS_UNTIL_MIN,
DAYS_UNTIL_MAX = 0, 60` to `feature_spec.py` as a distinct, correctly-named
bound, and using it (instead of the cycle-length bound) everywhere the
model's `days_until` output is clipped.

## About the training data

I don't have access to the real `menstrual_cycle_dataset.csv` this project
originally used (it's gitignored and wasn't in the repo I was given), so
`app/ml/generate_dataset.py` builds a **synthetic** dataset instead: ~600
simulated users across three archetypes (regular / irregular-PCOS-like /
stress-variable cycles), each with several simulated cycles and a random
"check-in day" per cycle, engineered so the label always depends on
information that would genuinely be available at that check-in.

This is clearly labelled as synthetic, not real clinical data — every
assumption is documented in the module docstring. The point isn't that the
synthetic numbers are clinically meaningful; it's that the **pipeline**
(feature contract, leakage prevention, grouped validation) is now correct
and will produce honestly-validated results on real data too. Swapping in
a real dataset just means pointing `train_model.py`'s `CSV_PATH` at it —
the schema it expects is documented at the top of `load_and_engineer()`.

## Files touched

- `app/ml/feature_spec.py` — new. Single source of truth for encodings/features.
- `app/ml/generate_dataset.py` — new. Synthetic data generator (documented assumptions).
- `app/ml/train_model.py` — rewritten. Leakage-free features, grouped validation.
- `app/ml/predictor.py` — rewritten. Drops the `cycle_length` argument entirely.
- `app/services/cycle_service.py` — updated call site (one line removed).
- `app/api/dashboard.py` — updated `/api/predict` call site, kept `cycle_length`
  as a backward-compatible optional API field that now maps to `avg_previous`.
- `app/schemas/schemas.py` — comment clarifying the legacy field's meaning.
- `app/api/auth.py` — now imports the shared `flash`/`get_flashes` from
  `app.utils.utils` instead of a second local copy (unrelated small cleanup
  found while reviewing this module).
- `models_ml/cycle_model.pkl` — removed (6.3MB, unreferenced anywhere in the code).
