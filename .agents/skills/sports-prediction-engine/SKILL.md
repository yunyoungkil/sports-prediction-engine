---
name: sports-prediction-engine
description: Architecture, engineering standards, evaluation rules, documentation-sync rule and roadmap for the sports-prediction-engine (Betman Proto scraping, market-correction AI, scorecard/CLV evaluation, data contracts, Streamlit dashboard, GitHub Actions + cron-job.org automation). Use when modifying any code in this repo, evaluating model performance, updating docs, or picking the next task from TODO.md.
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

## Documentation Sync Rule (MUST follow — automatic, no need to be asked)

Docs go stale silently (ARCHITECTURE.md described the v1 design for 5 days while 124 code commits landed). So **updating docs is part of the change, in the SAME commit**, whenever you notice any trigger below — do it without asking, and mention it in the summary.

| What you changed / learned | Update (same commit) |
|---|---|
| New, renamed or removed module in `collector/ models/ evaluation/ tracker/ notifier/ dashboard/` | `ARCHITECTURE.md` §3 module table (one line) |
| New or changed workflow, trigger, schedule, workflow step, external scheduler | `ARCHITECTURE.md` §1–2, `README.md` 자동 실행 section, `GEMINI.md` §7 |
| New or changed file in `data/` or its schema | `ARCHITECTURE.md` §4 (file table, and the `predictions.json` / `prediction_log` schema block if fields changed) and add/adjust the pandera schema in `evaluation/data_contracts.py` |
| Dashboard tab / filter / metric added or removed | `README.md` 대시보드 사용법 |
| A verdict, live number or decision moved (scorecard n/300, CLV, stale odds, calibration, paper-bet league, a Phase finished) | `README.md` 현재 상태 table (+ date), `GEMINI.md` §5, `TODO.md` progress table / item |
| New `REVIEW.md` R-item or `TODO.md` item/phase | `README.md` 문서 table (R-range, Phase range) |
| New rule, convention, secret, API limit, credential expiry date | `GEMINI.md` (and `SCHEDULER_TOKEN_EXPIRES` in `evaluation/briefing.py` for the token) |
| New skill rule or workflow for the assistant | this file |

How to check: `python -m evaluation.docs_sync` (also printed in the session-start briefing as "📚 문서 동기화"). It flags any module / workflow / `data/` file that `ARCHITECTURE.md` never names, and docs far behind the code. **Zero ⚠️ before every commit that adds or removes a file.** If the briefing shows ⚠️ at session start, fix the docs first, then continue the task.
Also: bump the "최종 갱신" date in `ARCHITECTURE.md`, keep numbers in README/GEMINI dated, and never write a number you did not read from a file or run output.

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
- **`evaluation/`**: `scorecard.py` (official pre-kickoff scorecard: LL + Brier/ECE, per-type and per-fixture counts, hypotheses H1/H2 via `league_group_v2`), `clv.py` (closing-odds comparison, fixture-clustered CI), `data_contracts.py` (pandera schemas + result-coverage check → `data/contract_report.json`), `health_check.py` (ops warnings → run marked failed), `briefing.py` (session-start briefing incl. scheduler, token expiry, data intake, contracts, docs sync), `docs_sync.py` (docs vs code check), `gates.py`, `stale_odds.py`, `walk_forward.py`, `market_scan.py`, `calibration_experiment.py`. Full catalog: `ARCHITECTURE.md` §3.
- **`tracker/`**: `prediction_log.py` (pre-kickoff AI probabilities for all AI-applied types; `first_*` fields are never overwritten), `odds_history.py` (per-round odds: first seen + changes only, written hourly by `results.yml` and by the main pipeline), `quick_results.py` (hourly job).
- **`notifier/telegram_bot.py`**: alerts via `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars.
- **`dashboard/app.py`**: Streamlit UI (predictions, parlay basket, past-match verification tab).
- **`run_pipeline.py`**: single entry point (collect → score past rounds → predict → EV → JSON → notify).
- **`.github/workflows/`**: `pipeline.yml` (full run: archive sync → weekly retrain when ≥6 days old → `run_pipeline.py` → data contracts → commit → ops verify), `results.yml` (numpy-only results/scorecard refresh), `test.yml`. Triggered by **cron-job.org via `workflow_dispatch`** (GitHub `schedule` is a delayed/dropped backup). See `ARCHITECTURE.md` §2, `GEMINI.md` §7.

## Known Issues (as of 2026-10-30)

Open items are tracked in `REVIEW.md` (R1~R20):

- **R6:** Telegram secrets not registered in GitHub (user action). `ODDS_API_KEY` is registered.
- **R14 resolved:** 홀/짝 is `category == "홀짝"`; 3-way logic still filters `game_type == "축구 승무패"` and draw odds > 1.
- **R17~R20 (AI model / data use audit, 2026-10-30):** no leakage found. R19 (line-aware grading/closing odds) and R20 (fixture counts beside the 300-game rule) fixed; R18 fixed for H1/H2 hypotheses; **R17/R18 model-feature fixes (`norm_team`, `league_group_v2`) are implemented but OFF** — switch on together with the `shin_mkt` replacement after the 300-game verdict (changing the model mid-sample would mix the verdict).
- K리그, J2, and the Chinese Super League are not covered by The Odds API.
- `match_state` can be 0 for the first hours after a new round opens (Betman API) — normal.
- `calculate_payout` floors at 1원. Betman's "원단위 절삭" may mean 10원 units (unverified).

## Roadmap & Workflow

- **Session start:** run `python -m evaluation.briefing` before the first reply and summarize it (repo sync, tests, verdict progress, health check). A SessionStart hook in the untracked `.claude/settings.local.json` does this on some PCs; where it is missing, run it manually. See `GEMINI.md` section 0.
- **v1 (ROADMAP.md Phase 1~5):** built, except Streamlit Cloud deployment.
- **v2 upgrade plan:** summarized in `README.md` (🚀 고도화 계획) and broken into checkboxes in **`TODO.md`**:
  - Step 0 무인 자동화 복구 → Phase 1 측정 기반 → Phase 2 모델 기본기 보강 → Phase 3 시장과 결합 → Phase 4 자금 관리와 제품 → **Phase 5 AI 정확도 향상** (market corrector, live sharp odds, weekly retraining, xG/lineups, more Betman leagues) → **Phase 6 우위 검증과 운영 안정화** (health check, stale-odds measurement, walk-forward backtest, per-market margin/edge table) → **Phase 7 표본 누적 후 예측 모델 업그레이드 실험** (post-300-sample walk-forward experiments)
- **Phase 6 (done 2026-10-27/29) principles still apply:** measure before building. Don't add models/features while samples are small (scorecard < 300). Stale-odds was measured (130 games → **no stale odds**, so no alerts/recommendations were built). Don't feed paper-bet results back into training. Don't add new Odds API calls.
- **Phase 7 roadmap (run AFTER live sample targets are reached):**
  1. **7.1 Non-linear margin removal (Power / Shin / piecewise slope):** test replacing proportional `remove_margin` to better capture Betman's heavy favorite-longshot bias (94.8% return on ≤1.3 favorites vs 59.2% on ≥5.0 longshots).
  2. **7.2 Two-stage `BetmanCorrector` + `sharp_diff` integration:** once `data/history/sharp_snapshots/` reaches ≥ 300 matched results, combine `log(p_sharp) - log(p_betman)` with `BetmanCorrector` and fall back to Betman-only features for unmapped leagues (K-League, J2).
  3. **7.3 xG / SOT overperformance ("Luck Index") correction:** refine the home-form overvaluation penalty using `actual goals - expected goals (xG/SOT)`.
  4. **7.4 Sport/market-specific features in `models/betman_markets.py`:** test handicap line value (`handi_val`) + league group + form for 3-way soccer handicap, back-to-back/rest features for basketball, and pre-game starting pitcher / lineup odds-lag checks for daytime KBO/K-League/J-League matches.
  - Every Phase 7 experiment MUST use strict chronological walk-forward (`models/betman_experiments.py` style) with cluster bootstrap CI, and only ship if it beats the current production baseline across most walk-forward windows.
- **Multi-PC sync & Future Launch (Feature B in `TODO.md`):**
  - Currently, personal selections/slips (`current_selections.json`, `my_slips.json`) are synced across the user's PCs via OneDrive using the `SPE_USER_DATA_DIR` environment variable (`C:\Users\<user>\OneDrive\spe_user_data`).
  - For future public/multi-user release (**Feature B** in `TODO.md`), add Supabase Auth + PostgreSQL (`user_id`-isolated RLS) behind `tracker/my_slips.py` and keep unauthenticated guests strictly in `st.session_state` so server disk files are never shared across users.
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
