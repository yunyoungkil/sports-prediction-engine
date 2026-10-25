---
name: sports-prediction-engine
description: Architecture, engineering standards, evaluation rules, and v2 upgrade roadmap for the sports-prediction-engine (Betman Proto scraping, Dixon-Coles/Poisson models, EV value picks, result tracking, Streamlit dashboard, GitHub Actions automation). Use when modifying any code in this repo, evaluating model performance, or picking the next task from TODO.md.
---

# Sports Prediction Engine Skill

This skill defines the architectural blueprint, realistic design constraints, evaluation rules, and upgrade roadmap for the Sports Match Outcome Prediction and Value Betting System (integrated with Betman Proto data).

## Core Principles (Realistic & Pragmatic Constraints)

1. **No Social/Comment NLP Scraping:**
   - DO NOT scrape YouTube comments, Betman forums, or community threads for sentiment analysis.
   - Reason: High noise, intense popularity bias (favoring famous teams), slang/meme parsing fragility, and poor correlation with true win probabilities.
   - Instead, prioritize structured numeric facts: Home/Away records, rest days (fatigue), shots/shots on target, and expected goals (xG).

2. **No Interactive Colab Dependency for Daily Automation:**
   - DO NOT rely on Google Colab for scheduled production runs (Colab requires manual execution).
   - Colab is ONLY for offline model research and experimental training.
   - Production inference runs inside **GitHub Actions** (CPU-based, lightweight models like `penaltyblog` Poisson/Dixon-Coles).

3. **Value Betting (EV) Over Blind Accuracy:**
   - Pure win rate prediction in 3-way soccer tops out around 50~56%. **Optimizing hit rate is NOT the goal** — live data showed S-grade picks hitting 53.7% while returning ROI −6.1%.
   - The primary objective is identifying **Positive Expected Value (EV > 0)**:
     $$\text{EV} = (\text{Model Probability} \times \text{Betman Odds}) - 1$$
   - Betman payout is ~85%, so a real edge must beat a large margin. Parlays compound the margin (0.85² ≈ 0.72).

4. **Market Is the Baseline:**
   - The margin-removed Betman probability (`models/odds_normalizer.remove_margin`) is the baseline model. A model change is only an improvement if it beats this baseline on held-out data.
   - Sharp bookmaker odds (Pinnacle, `PSH/PSD/PSA`, closing `PSCH/PSCD/PSCA` in football-data CSVs) are the best available "true probability" reference.

5. **100% Free Serverless Architecture:**
   - Pipeline: Betman Scraper -> Statistical Model (`penaltyblog`) -> GitHub Repository (JSON as DB) -> Streamlit Community Cloud UI -> Telegram Bot (optional).

## Evaluation Rules (MUST follow)

- **No lookahead:** Train only on data before the evaluated match date (strict chronological walk-forward).
- **Fixed train/validation split:** Tune constants (xi, form weights, λ scaling, grade thresholds) on the training period only. Report validation-period numbers only.
- **Report probability quality first:** Log loss, Brier score, and calibration — always side by side with the market baseline.
- **Report uncertainty:** Any ROI must come with a bootstrap 95% CI and sample size. Do not claim an edge from < 300 bets or from a CI that includes 0.
- **Track CLV:** Compare pick-time odds with closing odds; consistently positive CLV is the fastest signal of a real edge.
- **Never write in-sample backtest ROI into commit messages, README, or the dashboard as if it were a real result.** Live tracked stats live in `data/performance_stats.json`.

## Key Components

- **`collector/`**
  - `betman_scraper.py`: current round matches & odds via `gameScheduleDetlIFR.do` (session) + `buyPsblGame/detailSchedule.do` (`gmId=G101`).
  - `betman_results_scraper.py`: finished round results via `winrstDetl.do`.
  - `download_history.py`: football-data.co.uk CSVs for the top 5 European leagues → `data/history/all_historical_matches.csv`.
- **`models/`**
  - `poisson_model.py`: `MatchPredictor` — Dixon-Coles fit + per-category probabilities (승무패, 핸디캡, 언더오버, 전반전). Falls back to odds-derived λ when a team is unmapped or the sport is non-soccer.
  - `odds_normalizer.py`: proportional margin removal (2-way and 3-way).
  - `evaluator.py`: EV calculation and S/A/B/PASS grading.
  - `form_analyzer.py`: recent form (W-D-L) and H2H.
  - `parlay_builder.py`: 2~3 fold combinations, Banker + Wheel hedging.
- **`tracker/result_evaluator.py`**: scores past rounds and writes `data/performance_stats.json`.
- **`notifier/telegram_bot.py`**: alerts via `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars.
- **`dashboard/app.py`**: Streamlit UI (predictions, parlay basket, past-match verification tab).
- **`run_pipeline.py`**: single entry point (collect → score past rounds → predict → EV → JSON → notify).
- **`.github/workflows/pipeline.yml`**: scheduled run (Wed/Fri) that commits updated `data/` back to the repo.

## Known Issues (as of 2026-10)

Fix these before trusting automation output (tracked in `TODO.md` Step 0):

- Round number is hardcoded to `260126` (`run_pipeline.py`, `betman_scraper.py`, `betman_results_scraper.py`) — scheduled runs re-fetch the same round and fall back to cache.
- `data/history/betman_recent_results.json` is only fetched when missing, so live scoring is stuck at rounds 260118–260125.
- `pipeline.yml` does not pass Telegram secrets as `env`, so alerts never fire in Actions.
- `MatchPredictor._load_and_train` swallows training errors (`except Exception: pass`).
- Historical data stops at the 2024/25 season, has no time-decay weights, and uses one global model across leagues.
- Many constants (`* 1.15`, `+ 0.25`, `* 0.75`, `0.45`, `form_diff * 0.0015`) have no documented tuning source.

## Roadmap & Workflow

- **v1 (ROADMAP.md Phase 1~5):** built, except Streamlit Cloud deployment.
- **v2 upgrade plan:** summarized in `README.md` (🚀 고도화 계획) and broken into checkboxes in **`TODO.md`**:
  - Step 0 무인 자동화 복구 → Phase 1 측정 기반 → Phase 2 모델 기본기 보강 → Phase 3 시장과 결합 → Phase 4 자금 관리와 제품
- When picking up work: take the **first unchecked item in `TODO.md`**, do one item at a time, mark it `[x]`, and add before/after evaluation numbers to the progress table at the bottom.
- Respect the decision checkpoints in `TODO.md` (e.g., if the blended-model coefficient for our model ≈ 0, pivot to sharp-odds mispricing instead of model tuning).

## Coding Conventions

- Keep the Windows UTF-8 stdout guard (`sys.stdout.reconfigure(encoding='utf-8')`) at the top of runnable scripts.
- User-facing logs, docs, and dashboard text are in Korean; code identifiers in English.
- Data files are JSON/CSV under `data/` — keep schemas compatible with `ARCHITECTURE.md` and update it when a schema changes.
- Commit messages use conventional prefixes (`feat:`, `fix:`, `perf:`, `docs:`).
