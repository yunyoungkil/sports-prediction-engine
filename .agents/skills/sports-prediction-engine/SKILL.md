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
   - Production inference runs inside **GitHub Actions** (CPU-based, lightweight models (numpy/scipy logistic corrections)).

3. **Value Betting (EV) Over Blind Accuracy:**
   - Pure win rate prediction in 3-way soccer tops out around 50~56%. **Optimizing hit rate is NOT the goal** — live data showed S-grade picks hitting 53.7% while returning ROI −6.1%.
   - The primary objective is identifying **Positive Expected Value (EV > 0)**:
     $$\text{EV} = (\text{Model Probability} \times \text{Betman Odds}) - 1$$
   - Betman payout is ~85%, so a real edge must beat a large margin. Parlays compound the margin (0.85² ≈ 0.72).

4. **Market Is the Baseline:**
   - The margin-removed Betman probability (`models/odds_normalizer.remove_margin`) is the baseline model. A model change is only an improvement if it beats this baseline on held-out data.
   - Sharp bookmaker odds are the best available "true probability" reference. Historical: Pinnacle opening `PSH/PSD/PSA` (missing from 2526 on) and Betfair Exchange opening `BFEH/BFED/BFEA`. Live: The Odds API (`collector/betman_sharp.py`).
   - Closing odds (`PSC*`, `BFEC*`, `B365C*`) are lookahead. Use them only as a labeled upper bound, never as a feature or bet trigger.
   - Current approach (Phase 5): start from market probabilities and learn only corrections (`models/market_corrector.py`). From-scratch Dixon-Coles adds nothing over the market (blend b = 0).

5. **100% Free Serverless Architecture:**
   - Pipeline: Betman Scraper -> Market-correction AI (`models/betman_corrector.py`) -> GitHub Repository (JSON as DB) -> Streamlit Community Cloud UI -> Telegram Bot (optional).

## Evaluation Rules (MUST follow)

- **No lookahead:** Train only on data before the evaluated match date (strict chronological walk-forward).
- **Fixed 3-way split (`evaluation/config.py`):** TRAIN 2223–2324 → choose constants/regularization on TUNE 2425 → report TEST 2526 only (plus in-progress 2627 and Betman live results as extra held-out sets). A fitted coefficient must come from a model that did not train on the fit season.
- **Gates for live use:** `BetmanLivePredictor.load()` and `MarketCorrector.load()` return None (pipeline falls back to market probabilities) unless the saved params beat the market on held-out data. Keep this fallback.
- **Do not edit completion criteria in TODO.md** to make an item pass. Ask the user first.
- **Report probability quality first:** Log loss, Brier score, and calibration — always side by side with the market baseline.
- **Report uncertainty:** Any ROI must come with a bootstrap 95% CI and sample size. Do not claim an edge from < 300 bets or from a CI that includes 0.
- **Track CLV:** Compare pick-time odds with closing odds; consistently positive CLV is the fastest signal of a real edge.
- **Never write in-sample backtest ROI into commit messages, README, or the dashboard as if it were a real result.** Live tracked stats live in `data/performance_stats.json`.

## Key Components

- **`collector/`**
  - `betman_scraper.py`: current round matches & odds via `gameScheduleDetlIFR.do` (session) + `buyPsblGame/detailSchedule.do` (`gmId=G101`).
  - `betman_results_scraper.py`: finished round results via `winrstDetl.do`.
  - `download_history.py`: football-data.co.uk CSVs for the top 5 European leagues → `data/history/all_historical_matches.csv` (keeps B365, PS, BFE, Avg, Max odds; opening and closing).
  - `betman_sharp.py`: maps Betman leagues to The Odds API sport keys, 1 credit per league call, matches events by kickoff ±75 min / team-name similarity / odds distance (skips ambiguous), saves `data/history/sharp_snapshots/{gm_ts}.json`. Needs `ODDS_API_KEY`.
- **`models/`**
  - `market_view.py`: per-category display labels and margin-free market probabilities (replaced the removed Dixon-Coles `poisson_model.py`, which was worse than the market).
  - `odds_normalizer.py`: proportional margin removal (2-way and 3-way).
  - `evaluator.py`: EV calculation and S/A/B/PASS grading.
  - `form_analyzer.py`: recent form (W-D-L) and H2H.
  - `combo_builder.py`: Betman 맞춤조건구매 rules (odds product truncated to 2 decimals then ceil to 1 decimal).
  - `betman_markets.py`: Phase 5.11 per-type (segment = Betman game type × 2/3 outcomes) correction `log p = (1+β)·log p_market + c`, trained on `data/history/betman_markets/{yy}.csv.gz` (all sports/types, collected by `collector/betman_markets_archive.py`; labels = Betman's own GAME_RESULT code). Per-segment gate; `MarketsLivePredictor` applies only passing segments (as of 2026-10-26: 축구 핸디캡|3, 농구 승패|2) with `prob_source="betman_markets"`. Run `python -m models.betman_markets`.
  - `market_corrector.py`: Phase 5.1 market-offset multinomial logit. Feature sets `live` (used in production when gate passes), `live_sharp` (waiting for Betman validation), `full` (research). Run `python -m models.market_corrector`.
- **`evaluation/`**: `scorecard.py` (official pre-kickoff scorecard; grades every AI-applied type via Betman `game_result`, with `by_game_type`), `clv.py` (Phase 5.12 closing-line comparison → `data/clv_report.json`; evaluation only, never a feature), `sharp_vs_betman.py` (Phase 5.4 live validation, verdict at ≥ 300 matched results).
- **`tracker/`**: `prediction_log.py` (pre-kickoff AI probabilities for all AI-applied types; `first_*` fields are never overwritten), `odds_history.py` (per-round odds: first seen + changes only, written hourly by `results.yml` and by the main pipeline), `quick_results.py` (hourly job).
- **`notifier/telegram_bot.py`**: alerts via `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars.
- **`dashboard/app.py`**: Streamlit UI (predictions, parlay basket, past-match verification tab).
- **`run_pipeline.py`**: single entry point (collect → score past rounds → predict → EV → JSON → notify).
- **`.github/workflows/pipeline.yml`**: scheduled run twice daily (00:30 and 09:30 UTC; weekly retrain on the Wednesday morning run; overseas odds fetched at most once per 20h per round) with `contents: write`; the bot commits updated `data/` to `master`. Always `git pull --rebase` before starting work and before pushing.

## Known Issues (as of 2026-10-25)

Open items are tracked in `REVIEW.md`:

- **R6:** Telegram secrets not registered in GitHub (user action). `ODDS_API_KEY` is registered.
- **R14:** `betman_scraper.py` labels 홀/짝 (`game_type == "축구 SUM"`, draw odds 0) as category "일반 승무패" (133 of 266 in round 260126). Any 3-way logic must filter `game_type == "축구 승무패"` and draw odds > 1.
- K리그, J2, and the Chinese Super League are not covered by The Odds API, and are not in the historical training data.
- `calculate_payout` floors at 1원. Betman's "원단위 절삭" may mean 10원 units (unverified).

## Roadmap & Workflow

- **v1 (ROADMAP.md Phase 1~5):** built, except Streamlit Cloud deployment.
- **v2 upgrade plan:** summarized in `README.md` (🚀 고도화 계획) and broken into checkboxes in **`TODO.md`**:
  - Step 0 무인 자동화 복구 → Phase 1 측정 기반 → Phase 2 모델 기본기 보강 → Phase 3 시장과 결합 → Phase 4 자금 관리와 제품 → **Phase 5 AI 정확도 향상** (market corrector, live sharp odds, weekly retraining, xG/lineups, more Betman leagues) → **Phase 6 우위 검증과 운영 안정화** (health check, stale-odds measurement, walk-forward backtest, per-market margin/edge table)
- **Phase 6 principles:** measure before building. Don't add models/features while samples are small (scorecard 52/300, sharp odds 28/300). Don't build stale-odds alerts until 6.2 judges stale odds exist. Don't feed paper-bet results back into training. Don't add new Odds API calls — use the accumulated `sharp_snapshots`/`odds_history`.
- When picking up work: take the **first unchecked item in `TODO.md`**, do one item at a time, mark it `[x]`, and add before/after evaluation numbers to the progress table at the bottom.
- Respect the decision checkpoints in `TODO.md`. The Phase 3 checkpoint fired (b = 0), and that led to Phase 5.
- Collaboration: Gemini implements, Claude verifies and logs findings in `REVIEW.md` (R-numbers). Read `GEMINI.md` at the repo root for git, secret, and API-credit rules.
- The Odds API free tier is 500 credits/month. Don't call it in loops during development; cache responses and stub `collector.betman_sharp.fetch_league_events`.

## Coding Conventions

- Keep the Windows UTF-8 stdout guard (`sys.stdout.reconfigure(encoding='utf-8')`) at the top of runnable scripts.
- User-facing logs, docs, and dashboard text are in Korean; code identifiers in English.
- Data files are JSON/CSV under `data/` — keep schemas compatible with `ARCHITECTURE.md` and update it when a schema changes.
- Commit messages use conventional prefixes (`feat:`, `fix:`, `perf:`, `docs:`).
- Don't commit data files produced by local pipeline runs (`data/predictions.json`, `current_matches.json`, `performance_stats.json`, `sharp_validation.json`, `history/betman_recent_results.json`, `history/*_snapshots/`). The Actions bot owns them.
- Parse ISO dates without `dayfirst=True`.
