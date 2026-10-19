"""Simple dashboard: enter a 4th-down situation, see the recommendation.

Run with:  streamlit run app.py     (after `python train.py`)
"""
import streamlit as st

import config
from predict import describe_spot, load_bundle, recommend

st.set_page_config(page_title="NFL 4th Down Decision Predictor", page_icon="🏈")
st.title("🏈 NFL Fourth Down Decision Predictor")
st.caption("Trained on historical nflverse play-by-play data. The model learns what coaches "
           "*did* in similar spots, so it reflects real-world (often conservative) habits.")


@st.cache_resource
def get_models():
    try:
        return load_bundle("best_model"), load_bundle("conversion_model")
    except FileNotFoundError:
        return None, None


decision, conversion = get_models()
if decision is None:
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()

left, right = st.columns(2)
with left:
    yardline = st.slider("Yards from opponent end zone", 1, 99, 38,
                         help="38 = opponent's 38-yard line; 70 = your own 30")
    ydstogo = st.slider("Yards to go", 1, 25, 2)
    score_diff = st.number_input("Score difference (your score minus opponent)", -40, 40, -3)
with right:
    qtr = st.selectbox("Quarter", [1, 2, 3, 4, 5], index=3, format_func=lambda q: "OT" if q == 5 else f"Q{q}")
    minutes = st.number_input("Minutes left in quarter", 0, 15, 5)
    seconds = st.number_input("Seconds", 0, 59, 30)
    is_home = st.checkbox("Offense is the home team", value=True)

if st.button("Get recommendation", type="primary"):
    try:
        r = recommend(yardline, ydstogo, qtr, int(minutes), int(seconds), int(score_diff), is_home,
                      decision_bundle=decision, conversion_bundle=conversion)
    except ValueError as err:
        st.error(str(err))
    else:
        st.subheader(f"Recommendation: {r['recommendation']}")
        st.write(r["situation"])
        c1, c2, c3 = st.columns(3)
        c1.metric("P(go for it)", f"{r['p_go']:.0%}")
        if r["p_convert"] is not None:
            c2.metric("Est. conversion chance", f"{r['p_convert']:.0%}")
        c3.metric("Simple-rule baseline", r["rule_recommendation"])
