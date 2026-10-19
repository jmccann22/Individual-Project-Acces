# AI-Based NFL Fourth-Down Decision Predictor

Predicts whether an NFL offense should **go for it** or **kick** on 4th down, and compares
a Logistic Regression model and a Random Forest model against rule-based baselines. Data:
public nflverse play-by-play data (free, no login needed): https://github.com/nflverse/nflverse-data

## What's included

```
code/           the 6 Python files + requirements.txt
data/processed/ a 25-row sample of the cleaned data, so you can see the format
results/        the real results this code produces (summary.md, metrics.json, 2 figures)
Fourth_Down_Predictor_Colab.ipynb   same code, ready to run in Google Colab
```

## How to run it

**Easiest: Google Colab (no install needed)**
1. Go to colab.research.google.com and open a new notebook.
2. Upload `Fourth_Down_Predictor_Colab.ipynb` (File → Upload notebook).
3. Runtime → Run all.

**Or locally:**
```bash
cd code
pip install -r requirements.txt
python train.py
```
This downloads 6 seasons of data (first run only, needs internet), cleans it, and trains
both models. It writes `results/summary.md`, `results/metrics.json`, `results/figures/*.png`,
and `models/*.joblib`.

Then, to get one recommendation for a specific situation:
```bash
python predict.py --yardline 38 --ydstogo 2 --qtr 4 --clock 5:30 --score-diff -3 --home
```
(`--yardline` is yards from the opponent's end zone: 38 = opponent's 38-yard line.)

Or launch the dashboard version instead:
```bash
streamlit run app.py
```

## Versions used

Python 3.10+. See `code/requirements.txt` for package versions (pandas, numpy,
scikit-learn, matplotlib, joblib, pyarrow, streamlit). No GPU needed — runs fine on a
free Colab CPU runtime, under 10 minutes total (mostly the data download).

## AI assistance

Claude (Anthropic) was used to help write and debug this code and this README. The
results in `results/` are real output from actually running this code — not generated
or estimated by AI.
