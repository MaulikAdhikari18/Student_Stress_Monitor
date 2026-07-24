# Migration Guide — Monolith → Organized Structure

## What changed

Everything that used to live inside one ~2500-line `show_main_app()` function in
`app/dashboard.py` (plus a separate, simpler `app.py`) has been split into:

```
project/
├── app.py                    # Entrypoint — routing only (~30 lines)
├── requirements.txt
├── src/
│   ├── config.py             # LABELS, COLORS, EMOJIS, FEATURES, paths — single source of truth
│   ├── train_model.py         # unchanged logic, now imports from config instead of redefining it
│   ├── db/
│   │   ├── connection.py     # get_db()
│   │   ├── schema.py         # init_all() — creates all tables
│   │   ├── users.py          # auth: hash_password, create_user, verify_user
│   │   ├── sessions.py       # save_session, load_sessions
│   │   ├── goals.py          # load_goals, save_goals_db
│   │   └── tasks.py          # add_task, load_tasks, toggle_task, delete_task
│   ├── ml/
│   │   ├── scoring.py        # the interpretable 0-100 formula (was duplicated 3x)
│   │   └── model.py          # load_model, predict_stress (correction logic, was duplicated 2x)
│   ├── services/
│   │   ├── planner.py        # generate_weekly_schedule, get_break_schedule
│   │   ├── streaks.py        # compute_streaks, week_progress
│   │   └── report.py         # generate_stress_report (PDF)
│   └── ui/
│       ├── styles.py         # all CSS, as functions to call once per page
│       ├── components.py     # num_card, summary_card, level_badge, calendar colors
│       ├── landing_page.py
│       ├── auth_page.py
│       ├── main_app.py       # router: navbar + shared prediction state + page dispatch
│       └── pages/
│           ├── entry_page.py
│           ├── dashboard_page.py
│           ├── history_page.py
│           ├── summary_page.py
│           ├── goals_page.py
│           └── planner_page.py
├── data/
└── models/
```

## How to adopt it

1. Copy `app.py`, `requirements.txt`, and the whole `src/` folder into your project root.
2. Keep your existing `data/` and `models/` folders as-is — paths in `config.py` point
   to the same locations (`data/stress_monitor.db`, `models/model.pkl`, etc.), so your
   existing database and trained model keep working unchanged.
3. Archive or delete the **old** `app.py` and `app/dashboard.py` (the monoliths) — the
   new `app.py` at the project root replaces both. `app/dashboard.py` was treated as
   the canonical app; the old standalone `app.py` (simpler, no auth/DB) was not ported,
   since its functionality is now a subset of the main app.
4. Run `pip install -r requirements.txt`, then `streamlit run app.py` — behavior should
   be identical to before.

## Real bugs fixed during the split (not just moved code)

- **The stress-score formula was duplicated three times** (`app.py`, dashboard's New
  Entry page, dashboard's shared "latest session" block), with inconsistent handling
  of `exercise` units (0-7 weekly vs 0/1 daily). It's now one function,
  `ml/scoring.py::compute_stress_score()`, plus a `daily_exercise_to_weekly()` helper
  so the unit conversion happens in exactly one place.
- **The "ML prediction correction" logic was duplicated twice** with identical code.
  It's now `ml/model.py::predict_stress()`.

## Known follow-ups (flagged, not changed silently)

- **Password hashing is plain unsalted SHA-256** (`db/users.py`). Kept as-is so
  existing accounts keep working. If you want to move to bcrypt/passlib, that needs a
  one-time migration for existing users — happy to do that as a separate step.
- **Data folder has multiple stale/overlapping CSVs** (`student_stress.csv`,
  `student_stress_large.csv`, `student_stress_data.csv`, `stress_data.csv`,
  `goals.csv`) left over from notebook experimentation. Only `student_stress_data.csv`
  (14-feature schema) is used by `train_model.py` / the app. Worth deleting the rest
  once you confirm you don't need them.
- The original `README.md` is still the generic template with `[INSERT ...]`
  placeholders — worth a rewrite now that the structure is settled.

## Verification performed on this refactor

- All 30 files pass `ast.parse` (syntax check).
- `db/`, `ml/`, `services/` modules were actually imported and exercised with real
  logic (score computation, ML correction fallback, PDF generation, schedule
  generation, streak counting) — not just syntax-checked.
- All six `ui/pages/*.render()` functions, plus the `main_app.show_main_app()` router,
  were called end-to-end against synthetic session data using a Streamlit stub (since
  this sandbox has no network access to install the real `streamlit` package). This
  catches typos, wrong argument counts, and undefined variables that pure syntax
  checking would miss — but it does **not** replace running the real app with
  `streamlit run app.py` in your own environment, which you should still do before
  trusting this in place of the original.
