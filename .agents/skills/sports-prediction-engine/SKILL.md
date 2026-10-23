---
name: sports-prediction-engine
description: Sports match outcome prediction and value betting pipeline guide and engineering standards for sports-prediction-engine.
---

# Sports Prediction Engine Skill

This skill defines the architectural blueprint, realistic design constraints, and step-by-step roadmap for developing the Sports Match Outcome Prediction and Value Betting System (specifically integrated with Betman Proto data).

## Core Principles (Realistic & Pragmatic Constraints)

1. **No Social/Comment NLP Scraping:**
   - DO NOT scrape YouTube comments, Betman forums, or community threads for sentiment analysis.
   - Reason: High noise, intense popularity bias (favoring famous teams), slang/meme parsing fragility, and poor correlation with true win probabilities.
   - Instead, prioritize structured numeric facts: Home/Away records, rest days (fatigue), and expected goals (xG).

2. **No Interactive Colab Dependency for Daily Automation:**
   - DO NOT rely on Google Colab for scheduled production runs (Colab requires manual execution).
   - Colab is ONLY for offline model research and experimental training.
   - Production inference runs inside **GitHub Actions** (CPU-based, finishes in < 15 seconds using lightweight models like `penaltyblog` Poisson/Dixon-Coles or pre-trained `LightGBM`).

3. **Value Betting (EV) Over Blind Accuracy:**
   - Pure win rate prediction in 3-way soccer tops out around 50~56%.
   - The primary objective is identifying **Positive Expected Value (EV > 0)**:
     $$\text{EV} = (\text{Model Probability} \times \text{Betman Odds}) - 1$$
   - Recommended picks are games where Betman's implied probability is significantly lower than the model's computed probability.

4. **100% Free Serverless Architecture:**
   - Pipeline: Betman Data Scraper -> Statistical Model (`penaltyblog`) -> GitHub Repository -> Streamlit Community Cloud UI -> Telegram Bot (optional).

## Key Components

- **`collector/`**: Betman scraping scripts (`scraper.py`) targeting `gameSlip.do` & AJAX endpoints (`gmId=G101`).
- **`models/`**: Statistical models (`poisson_model.py`, Dixon-Coles, Elo rating, odds margin removal).
- **`dashboard/`**: Streamlit application (`app.py`) for viewing predictions, probabilities, and value picks.
- **`.github/workflows/`**: Scheduled GitHub Actions runner (`run_pipeline.yml`).
