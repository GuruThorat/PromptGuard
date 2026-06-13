"""PromptGuard live dashboard (Streamlit).

Reads the SQLite request log the proxy writes to and shows traffic, verdicts, block
rate, and latency in real time.

Run:  .venv/bin/streamlit run dashboard/app.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402

st.set_page_config(page_title="PromptGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ PromptGuard — LLM Firewall")
st.caption("Real-time prompt-injection / jailbreak screening in front of a local LLM.")


@st.cache_data(ttl=2)
def load() -> pd.DataFrame:
    if not config.DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        return pd.read_sql_query("SELECT * FROM requests ORDER BY id DESC", con)
    finally:
        con.close()


df = load()
if df.empty:
    st.info("No traffic yet. Start the proxy and POST a prompt to /chat, "
            "then reload this page.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total prompts", len(df))
c2.metric("Blocked", int(df.blocked.sum()))
c3.metric("Block rate", f"{df.blocked.mean():.0%}")
c4.metric("Median latency", f"{df.latency_ms.median():.0f} ms")

left, right = st.columns([1, 2])
with left:
    st.subheader("Verdicts")
    st.bar_chart(df.verdict.value_counts())
with right:
    st.subheader("Malicious-score distribution")
    st.bar_chart(pd.cut(df.score, bins=[0, .2, .4, .6, .8, 1.0]).value_counts().sort_index())

st.subheader("Recent traffic")
show = df[["ts", "verdict", "score", "blocked", "latency_ms", "source", "prompt"]].head(60).copy()
show["prompt"] = show["prompt"].str.slice(0, 120)


def _row_style(row):
    bg = "background-color: #5b1a1a" if row.blocked else "background-color: #14361f"
    return [bg] * len(row)


st.dataframe(show.style.apply(_row_style, axis=1), use_container_width=True, height=460)
st.caption("Cache TTL 2s — reload to refresh.")
