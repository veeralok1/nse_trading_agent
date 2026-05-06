"""
app.py — NSE Intraday Trading Agent · Streamlit UI
Run: streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
      background: #1e2130;
      border-radius: 10px;
      padding: 16px;
      text-align: center;
  }
  .buy-badge  { background:#0a3d0a; color:#00e676; padding:4px 12px;
                border-radius:20px; font-weight:bold; }
  .sell-badge { background:#3d0a0a; color:#ff5252; padding:4px 12px;
                border-radius:20px; font-weight:bold; }
  .hold-badge { background:#3d3a0a; color:#ffeb3b; padding:4px 12px;
                border-radius:20px; font-weight:bold; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────────────────
# CACHED ORCHESTRATOR & LLM AGENT
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Initialising trading agents…")
def get_orchestrator():
    from orchestrator import TradingOrchestrator
    return TradingOrchestrator(use_cache=True)


@st.cache_resource(show_spinner="🤖 Initialising AI agent…")
def get_llm_agent():
    from agents.llm_agent import LLMAgent
    return LLMAgent()


orch = get_orchestrator()
llm  = get_llm_agent()


# ─────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────

CHART_THEME = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#c0c0c0"),
)
GRID_COLOR = "#1e2130"


def make_candlestick_chart(df: pd.DataFrame, symbol: str, show_indicators: List[str]) -> go.Figure:
    """
    Build a multi-panel candlestick chart with:
    Panel 1 (70%): OHLC + MA overlays + BB + VWAP + signals
    Panel 2 (15%): Volume
    Panel 3 (15%): RSI or MACD
    """
    rows    = 3
    heights = [0.65, 0.15, 0.20]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
        subplot_titles=("", "Volume", "RSI / MACD"),
    )

    # ── Panel 1: Candlestick ─────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff5252",
        ),
        row=1, col=1,
    )

    # Bollinger Bands
    if "BB_upper" in df.columns and "BB Bands" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,100,255,0.4)", dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(100,100,255,0.4)", dash="dot"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.05)", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_mid"], name="BB Mid",
            line=dict(color="rgba(100,100,255,0.6)", dash="dash", width=1), showlegend=False,
        ), row=1, col=1)

    # Moving Averages
    ma_cfg = [
        ("MA_20", "#FFA726", "MA 20"),
        ("MA_50", "#42A5F5", "MA 50"),
        ("MA_200", "#EF5350", "MA 200"),
    ]
    if "Moving Averages" in show_indicators:
        for col, color, label in ma_cfg:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=label,
                    line=dict(color=color, width=1.2),
                ), row=1, col=1)

    # VWAP
    if "VWAP" in df.columns and "VWAP" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#E040FB", width=1.5, dash="dashdot"),
        ), row=1, col=1)

    # EMA 9 / 21
    if "EMA 9/21" in show_indicators:
        for col, color, label in [("EMA_9", "#66BB6A", "EMA 9"), ("EMA_21", "#FFCA28", "EMA 21")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=label,
                    line=dict(color=color, width=1),
                ), row=1, col=1)

    # ── Panel 2: Volume ──────────────────────────────────
    colors = ["#00e676" if c >= o else "#ff5252"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)

    if "Volume_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_MA20"], name="Vol MA20",
            line=dict(color="#FFA726", width=1), showlegend=False,
        ), row=2, col=1)

    # ── Panel 3: RSI ─────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#7C4DFF", width=1.5),
        ), row=3, col=1)
        # Overbought / Oversold bands
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,82,82,0.5)",
                      row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,230,118,0.5)",
                      row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                      row=3, col=1)

    # ── Layout ───────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"📈 {symbol.replace('.NS','')}", font=dict(size=18, color="#fff")),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=10),
        ),
        height=700,
        margin=dict(l=0, r=10, t=60, b=0),
        **CHART_THEME,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        showspikes=True, spikecolor="#888", spikesnap="cursor",
    )
    fig.update_yaxes(gridcolor=GRID_COLOR)

    return fig


def _badge(action: str) -> str:
    cls = {"BUY": "buy-badge", "SELL": "sell-badge"}.get(action, "hold-badge")
    return f'<span class="{cls}">{action}</span>'


def _color(action: str) -> str:
    return {"BUY": "#00e676", "SELL": "#ff5252", "HOLD": "#ffeb3b"}.get(action, "#fff")


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/"
             "NSE_Logo.svg/200px-NSE_Logo.svg.png",
             width=140, use_container_width=False)
    st.markdown("## ⚙️ Settings")

    selected_symbol = st.text_input(
        "Stock Symbol", value="RELIANCE",
        placeholder="e.g. RELIANCE, TCS, INFY",
        help="Enter NSE ticker or company name",
    ).upper().strip()

    interval_map = {"1 min": "1m", "5 min": "5m", "15 min": "15m",
                    "30 min": "30m", "1 Hour": "1h", "Daily": "1d"}
    interval_label = st.selectbox("Interval", list(interval_map.keys()), index=1)
    interval       = interval_map[interval_label]

    period_map = {"Today": "1d", "2 Days": "2d", "5 Days": "5d",
                  "1 Month": "1mo", "3 Months": "3mo"}
    period_label = st.selectbox("Period", list(period_map.keys()), index=1)
    period       = period_map[period_label]

    st.divider()
    st.markdown("**Chart Overlays**")
    show_indicators = st.multiselect(
        "Show Indicators",
        ["BB Bands", "Moving Averages", "VWAP", "EMA 9/21"],
        default=["Moving Averages", "VWAP"],
    )

    st.divider()
    universe = st.selectbox("Screening Universe", ["nifty50", "niftybank", "midcap"], index=0)
    top_n    = st.slider("Top N stocks", 3, 20, 10)

    st.divider()
    capital = st.number_input("Capital (₹)", value=100_000, step=10_000, format="%d")
    orch.ra.cfg.capital = float(capital)

    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        time.sleep(60)
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}")


# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 Analysis", "🏆 Top Stocks", "💬 Chat", "📉 Backtest", "📡 Screener"
])


# ═══════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ═══════════════════════════════════════════════════════

with tabs[0]:
    col_btn, col_refresh = st.columns([1, 5])
    with col_btn:
        run_analysis = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if run_analysis or "analysis_result" not in st.session_state:
        with st.spinner(f"Fetching data for {selected_symbol}…"):
            result = orch.analyze(selected_symbol, interval=interval, period=period)
            st.session_state["analysis_result"] = result
            st.session_state["analysis_symbol"]   = selected_symbol
    else:
        result = st.session_state.get("analysis_result", {})

    if "error" in result:
        st.error(result["error"])
    else:
        sym      = result["symbol"]
        quote    = result["quote"]
        ind_sum  = result["indicators"]
        trend    = result["trend"]
        signals  = result["signals"]
        risk     = result["risk_profile"]
        df       = result["df"]
        action   = result["action"]
        conf     = result["confidence"]

        # ── Quote row ─────────────────────────────────
        st.markdown(f"### {sym.replace('.NS','')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        price      = quote.get("last_price", 0)
        chg        = quote.get("change", 0)
        chg_pct    = quote.get("change_pct", 0)

        c1.metric("💰 LTP", f"₹{price:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
        c2.metric("📈 Trend", trend)
        c3.metric("🔔 Signal", action, f"{conf:.0%} confidence")
        c4.metric("📊 RSI", f"{ind_sum.get('rsi') or '—'}")
        c5.metric("📉 ATR", f"{ind_sum.get('atr') or '—'}")

        # ── Candlestick chart ─────────────────────────
        if not df.empty:
            fig = make_candlestick_chart(df, sym, show_indicators)
            st.plotly_chart(fig, use_container_width=True)

        # ── Signals + Risk in two columns ─────────────
        left, right = st.columns(2)

        with left:
            st.markdown("#### 🔔 Signals")
            if signals:
                for s in signals:
                    with st.container(border=True):
                        st.markdown(
                            f"**{s.action}** — `{s.strategy}` ({s.confidence:.0%})",
                            unsafe_allow_html=True,
                        )
                        for r in s.reasons:
                            st.markdown(f"- {r}")
            else:
                st.info("No strong signals at this time. **HOLD / Monitor.**")

        with right:
            st.markdown("#### 🛡️ Risk Profile")
            if risk:
                cols = st.columns(3)
                cols[0].metric("Entry",    f"₹{risk.entry_price:,.2f}")
                cols[1].metric("Stop Loss", f"₹{risk.stop_loss:,.2f}",
                               f"-{risk.risk_pct:.1f}%", delta_color="inverse")
                cols[2].metric("R:R Ratio", f"1:{risk.risk_reward:.1f}")

                st.markdown(
                    f"**Targets:** T1 ₹{risk.target_1:,.2f}  |  "
                    f"T2 ₹{risk.target_2:,.2f}  |  T3 ₹{risk.target_3:,.2f}"
                )
                st.markdown(
                    f"**Position Size:** {risk.position_size} shares  "
                    f"(Risk ₹{risk.risk_amount_inr:,.0f})"
                )
                st.caption(risk.notes)
            else:
                st.info("No trade setup. Wait for a signal.")

        # ── Indicator table ────────────────────────────
        with st.expander("📋 All Indicators"):
            ind_items = {
                "RSI":          ind_sum.get("rsi"),
                "MACD":         ind_sum.get("macd"),
                "MACD Signal":  ind_sum.get("macd_signal"),
                "MACD Hist":    ind_sum.get("macd_hist"),
                "MA 20":        ind_sum.get("ma_20"),
                "MA 50":        ind_sum.get("ma_50"),
                "MA 200":       ind_sum.get("ma_200"),
                "BB Upper":     ind_sum.get("bb_upper"),
                "BB Mid":       ind_sum.get("bb_mid"),
                "BB Lower":     ind_sum.get("bb_lower"),
                "BB %B":        ind_sum.get("bb_pct_b"),
                "VWAP":         ind_sum.get("vwap"),
                "ATR":          ind_sum.get("atr"),
                "Volume Ratio": ind_sum.get("volume_ratio"),
                "Momentum 5":   ind_sum.get("momentum_5"),
            }
            ind_df = pd.DataFrame(
                {"Indicator": list(ind_items.keys()),
                 "Value": [f"{v:.4f}" if v is not None else "—" for v in ind_items.values()]}
            )
            st.dataframe(ind_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — TOP STOCKS
# ═══════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("## 🏆 Top Stocks Screener")

    # ── Mode selector + controls ──────────────────────────
    mode_col, ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2, 3])
    with mode_col:
        screen_mode = st.radio(
            "Mode", ["📈 Swing / Investment", "⚡ Intraday"],
            index=0, horizontal=False,
            help="Swing uses daily charts (positional). Intraday uses 5-min charts.",
        )
    with ctrl1:
        rank_btn = st.button("🚀 Run Screener", type="primary", use_container_width=True)
    with ctrl2:
        action_filt = st.selectbox("Filter by Signal", ["All", "BUY only", "SELL only"], index=0)
    with ctrl3:
        if "Swing" in screen_mode:
            horizon_opt = st.selectbox(
                "Horizon",
                ["1 Month (30d)", "3 Months (90d)", "6 Months (180d)", "1 Year (365d)"],
                index=1,
            )
            horizon_map = {
                "1 Month (30d)":    "30d",
                "3 Months (90d)":   "90d",
                "6 Months (180d)":  "180d",
                "1 Year (365d)":    "365d",
            }
            swing_period = horizon_map[horizon_opt]
            st.caption("📅 Daily charts | Full Nifty50 universe | ~15–30 s")
        else:
            swing_period = None
            st.caption("⚡ 5-min charts | Top 25 Nifty50 | ~30–60 s on first run")

    # Only run when button is clicked
    if rank_btn:
        is_swing = "Swing" in screen_mode
        spinner_msg = (
            f"📅 Running swing screener ({swing_period})…"
            if is_swing else
            "⚡ Running intraday screener (5-min bars)…"
        )
        with st.spinner(spinner_msg):
            try:
                if is_swing:
                    from config import NIFTY_50_SYMBOLS
                    ranked = orch.rk.rank_swing(
                        NIFTY_50_SYMBOLS[:50], period=swing_period, top_n=top_n
                    )
                else:
                    ranked = orch.get_top_intraday(universe=universe, top_n=top_n)

                st.session_state["ranked"]      = ranked
                st.session_state["ranked_mode"] = screen_mode
                if not ranked:
                    st.session_state["ranked_error"] = (
                        "No stocks passed the filters. "
                        + ("Try expanding the universe or a different horizon."
                           if is_swing else
                           "This can happen outside NSE market hours (9:15–15:30 IST). "
                           "Try again in a minute.")
                    )
                else:
                    st.session_state.pop("ranked_error", None)
            except Exception as e:
                st.session_state["ranked_error"] = str(e)
                st.session_state["ranked"] = []

    ranked = st.session_state.get("ranked", None)
    err    = st.session_state.get("ranked_error", None)
    last_mode = st.session_state.get("ranked_mode", "")

    if ranked is None:
        st.info(
            "👆 Select a mode and click **Run Screener**.\n\n"
            "- **Swing / Investment** → daily chart analysis for positional trades\n"
            "- **Intraday** → 5-min chart analysis for same-day trades"
        )

    elif err:
        st.error(f"❌ Screener error: {err}")

    elif len(ranked) == 0:
        st.warning(
            "⚠️ No stocks returned.\n\n"
            "- For **Intraday**: market must be open (9:15 AM – 3:30 PM IST, Mon–Fri)\n"
            "- For **Swing**: try a shorter horizon like 1 Month\n"
            "- Yahoo Finance may have rate-limited — wait 60 s and retry"
        )

    else:
        rank_df = orch.rk.to_dataframe(ranked)

        # Apply signal filter
        display_df = rank_df.copy()
        if action_filt == "BUY only":
            display_df = display_df[display_df["Signal"] == "BUY"]
        elif action_filt == "SELL only":
            display_df = display_df[display_df["Signal"] == "SELL"]

        mode_label = "📅 Swing/Positional" if "Swing" in last_mode else "⚡ Intraday"
        st.success(
            f"✅ **{len(ranked)}** stocks found  |  Mode: {mode_label}  "
            f"|  Showing: **{len(display_df)}** after filter"
        )

        def _style_signal(val):
            colors = {"BUY":  "color: #00e676; font-weight:bold",
                      "SELL": "color: #ff5252; font-weight:bold",
                      "HOLD": "color: #ffeb3b"}
            return colors.get(val, "")

        if display_df.empty:
            st.warning(f"No stocks match the **{action_filt}** filter. Try **All**.")
        else:
            styled = display_df.style.map(_style_signal, subset=["Signal"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Score breakdown — only shown when display_df has data
        if not display_df.empty:
            shown_syms = set(display_df["Symbol"].tolist())
            chart_ranks = [r for r in ranked if r.symbol.replace(".NS", "") in shown_syms][:10]

            if chart_ranks:
                score_label = "Trend" if "Swing" in last_mode else "Volatility"
                st.markdown(f"#### 📊 Score Breakdown  _{score_label} · Liquidity · Momentum · Signal_")
                fig2 = go.Figure()
                score_cols   = ["volatility_score", "liquidity_score", "momentum_score", "signal_score"]
                score_labels = [score_label, "Liquidity", "Momentum", "Signal"]
                colors_bar   = ["#FFA726", "#42A5F5", "#66BB6A", "#7C4DFF"]

                top_syms = [r.symbol.replace(".NS", "") for r in chart_ranks]
                for col, label, color in zip(score_cols, score_labels, colors_bar):
                    vals = [getattr(r, col) for r in chart_ranks]
                    fig2.add_trace(go.Bar(name=label, x=top_syms, y=vals, marker_color=color))

                fig2.update_layout(
                    barmode="stack", height=350,
                    legend=dict(orientation="h", y=1.1),
                    **CHART_THEME,
                )
                fig2.update_xaxes(gridcolor=GRID_COLOR)
                fig2.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 3 — CHAT
# ═══════════════════════════════════════════════════════

with tabs[2]:
    # ── Header with AI status badge ───────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown("## 💬 Ask the Trading Agent")
    with col_badge:
        if llm.is_available:
            st.success(f"🤖 AI ON\n`{llm.model_name.split('/')[-1]}`")
        else:
            st.warning("🔒 AI OFF\nRule-based only")

    # ── Debug expander (shows secret resolution — remove after confirmed working)
    with st.expander("🔧 AI Config Debug", expanded=False):
        from config import ENABLE_LLM_CHAT, LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL
        st.code(
            f"ENABLE_LLM_CHAT  = {ENABLE_LLM_CHAT}\n"
            f"LLM_PROVIDER     = '{LLM_PROVIDER}'\n"
            f"OPENROUTER_MODEL = '{OPENROUTER_MODEL}'\n"
            f"API key present  = {bool(OPENROUTER_API_KEY)}\n"
            f"API key prefix   = '{OPENROUTER_API_KEY[:12]}...' "
            f"(len={len(OPENROUTER_API_KEY)})"
            if OPENROUTER_API_KEY else
            f"ENABLE_LLM_CHAT  = {ENABLE_LLM_CHAT}\n"
            f"LLM_PROVIDER     = '{LLM_PROVIDER}'\n"
            f"API key present  = False  ← this is the problem\n"
            f"\nMake sure OPENROUTER_API_KEY is set in Streamlit Secrets.",
            language="text",
        )
        if not llm.is_available:
            st.info(
                "**To enable AI:**\n"
                "1. Go to Streamlit Cloud → your app → ⋮ → Settings → Secrets\n"
                "2. Add:\n"
                "```toml\n"
                'OPENROUTER_API_KEY = "sk-or-v1-your-key"\n'
                'ENABLE_LLM_CHAT = "true"\n'
                'LLM_PROVIDER = "openrouter"\n'
                "```\n"
                "3. Save → app reboots automatically"
            )

    if "chat_history" not in st.session_state:
        welcome = (
            "👋 Hi! I'm your NSE Trading Agent. Ask me anything:\n\n"
            "- `Analyze RELIANCE`\n"
            "- `Is TCS good for intraday?`\n"
            "- `Top 5 stocks for today`\n"
            "- `What is the trend of HDFC Bank?`\n"
            "- `Price of INFY`"
        )
        if llm.is_available:
            welcome += f"\n\n✨ _AI analysis powered by **{llm.model_name}** via OpenRouter_"
        st.session_state.chat_history = [{"role": "assistant", "content": welcome}]

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if user_msg := st.chat_input("Ask about any NSE stock…"):
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                # Step 1: rule-based analysis (always runs)
                result        = orch.chat(user_msg)
                response_text = result["text"]
                data          = result.get("data", {})
                df_resp       = data.get("df")
                sym_resp      = result.get("symbol")
                intent_resp   = result.get("intent", "ANALYZE")
                horizon_resp  = data.get("horizon", "")

                st.markdown(response_text)

                # Step 2: AI enhancement (if LLM is enabled)
                # Works for both single-stock queries AND list queries (swing/intraday best)
                if llm.is_available:
                    # Build a meaningful LLM prompt regardless of intent
                    if intent_resp in ("SWING_BEST", "INTRADAY_BEST"):
                        # Summarise the ranked list for the LLM
                        stocks_data = data.get("stocks", [])
                        if stocks_data:
                            horizon_label = {
                                "30d": "1 month", "90d": "3 months",
                                "180d": "6 months", "365d": "1 year",
                            }.get(horizon_resp, "swing")
                            mode_label = (
                                f"{horizon_label} investment/positional"
                                if intent_resp == "SWING_BEST" else "intraday day-trading"
                            )
                            top_list = ", ".join(
                                f"{r.symbol.replace('.NS','')} ({r.action} {r.confidence:.0%})"
                                for r in stocks_data[:5]
                            )
                            llm_q = (
                                f"The user asked: \"{user_msg}\"\n\n"
                                f"Our rule-based screener returned these top stocks for {mode_label}: "
                                f"{top_list}.\n\n"
                                f"In 100–150 words, comment on why these stocks might be suitable "
                                f"for {mode_label} based on their signals and general market context. "
                                f"Remind the user this is educational only."
                            )
                            with st.spinner(f"🤖 AI commentary from {llm.model_name}…"):
                                ai_reply = llm.chat(llm_q)
                            st.markdown("---")
                            st.markdown(f"**🤖 AI Commentary ({llm.model_name.split('/')[-1]}):**")
                            st.markdown(ai_reply)
                            response_text += f"\n\n---\n**AI:** {ai_reply}"

                    elif sym_resp:
                        # Single-stock analysis enhancement
                        with st.spinner(f"🤖 Getting AI analysis from {llm.model_name}…"):
                            ai_reply = llm.chat(
                                user_msg,
                                context={
                                    "symbol":       sym_resp,
                                    "quote":        data.get("quote", {}),
                                    "indicators":   data.get("indicators", {}),
                                    "trend":        data.get("trend", ""),
                                    "signals":      data.get("signals", []),
                                    "risk_profile": data.get("risk_profile"),
                                },
                            )
                        st.markdown("---")
                        st.markdown(f"**🤖 AI Analysis ({llm.model_name.split('/')[-1]}):**")
                        st.markdown(ai_reply)
                        response_text += f"\n\n---\n**AI:** {ai_reply}"

                # Show chart if available (single-stock only)
                if df_resp is not None and not df_resp.empty and sym_resp:
                    mini_fig = make_candlestick_chart(df_resp, sym_resp, ["Moving Averages", "VWAP"])
                    mini_fig.update_layout(height=400)
                    st.plotly_chart(mini_fig, use_container_width=True)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response_text}
                )

    c1, c2 = st.columns([1, 5])
    if c1.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    if not llm.is_available:
        c2.caption(
            "💡 To enable AI: add `OPENROUTER_API_KEY=sk-or-v1-...` "
            "and `ENABLE_LLM_CHAT=true` to your `.env` file."
        )


# ═══════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("## 📉 Strategy Backtester")

    bc1, bc2, bc3 = st.columns(3)
    bt_symbol   = bc1.text_input("Symbol", value="RELIANCE")
    bt_strategy = bc2.selectbox(
        "Strategy",
        ["RSI Reversal", "MACD Crossover", "ORB Breakout", "MA Crossover", "BB Squeeze"],
    )
    bt_period = bc3.selectbox("Backtest Period", ["30d", "60d", "90d"], index=1)

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running backtest…"):
            from backtesting.backtester import Backtester, STRATEGY_MAP

            strategy_key = {
                "RSI Reversal":  "rsi",
                "MACD Crossover":"macd",
                "ORB Breakout":  "orb",
                "MA Crossover":  "ma_cross",
                "BB Squeeze":    "bb_squeeze",
            }[bt_strategy]

            bt = Backtester(orch.da, orch.ia)
            report = bt.run(bt_symbol, strategy=strategy_key, period=bt_period)

            if "error" in report:
                st.error(report["error"])
            else:
                # ── Metrics ──────────────────────────
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Trades",    report["total_trades"])
                m2.metric("Win Rate",        f"{report['win_rate']:.1%}")
                m3.metric("Total Return",    f"{report['total_return']:.2%}",
                          delta_color="normal")
                m4.metric("Avg Win",         f"{report['avg_win']:.2%}")
                m5.metric("Avg Loss",        f"{report['avg_loss']:.2%}",
                          delta_color="inverse")

                m6, m7, m8 = st.columns(3)
                m6.metric("Profit Factor",   f"{report['profit_factor']:.2f}")
                m7.metric("Max Drawdown",    f"{report['max_drawdown']:.2%}",
                          delta_color="inverse")
                m8.metric("Sharpe Ratio",    f"{report['sharpe']:.2f}")

                # ── Equity curve ─────────────────────
                equity_df = report.get("equity_curve")
                if equity_df is not None and not equity_df.empty:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=equity_df.index, y=equity_df["equity"],
                        fill="tozeroy", name="Equity",
                        line=dict(color="#00e676", width=2),
                        fillcolor="rgba(0,230,118,0.1)",
                    ))
                    fig_eq.update_layout(
                        title="Equity Curve", height=300, **CHART_THEME
                    )
                    fig_eq.update_xaxes(gridcolor=GRID_COLOR)
                    fig_eq.update_yaxes(gridcolor=GRID_COLOR)
                    st.plotly_chart(fig_eq, use_container_width=True)

                # ── Trade log ────────────────────────
                trade_df = report.get("trades")
                if trade_df is not None and not trade_df.empty:
                    with st.expander("📋 Trade Log"):
                        st.dataframe(trade_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# TAB 5 — SCREENER
# ═══════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("## 📡 Signal Screener")
    sc1, sc2 = st.columns([2, 3])
    scr_action = sc1.radio("Filter by Signal", ["All", "BUY", "SELL"], horizontal=True)
    scr_btn    = sc2.button("🔍 Screen Now", type="primary")

    # Only run when button is explicitly clicked
    if scr_btn:
        with st.spinner("📡 Scanning universe for active signals…"):
            try:
                action_f = None if scr_action == "All" else scr_action
                results  = orch.screen_signals(universe=universe, action_filter=action_f)
                st.session_state["screener_result"] = results
                st.session_state.pop("screener_error", None)
            except Exception as e:
                st.session_state["screener_error"] = str(e)
                st.session_state["screener_result"] = []

    results = st.session_state.get("screener_result", None)
    scr_err = st.session_state.get("screener_error", None)

    if results is None:
        st.info("👆 Click **Screen Now** to scan for live signals across the selected universe.")
    elif scr_err:
        st.error(f"❌ {scr_err}")
    elif not results:
        st.warning(
            "⚠️ No active signals found right now.\n\n"
            "Try during NSE market hours (9:15 AM – 3:30 PM IST, Mon–Fri) "
            "for best results."
        )
    else:
        st.success(f"✅ Found **{len(results)}** stocks with active signals")
        scr_df = pd.DataFrame(results)
        scr_df["confidence"] = scr_df["confidence"].apply(lambda x: f"{x:.0%}")
        scr_df["price"]      = scr_df["price"].apply(lambda x: f"₹{x:,.2f}")
        scr_df.columns       = ["Symbol", "Signal", "Strategy", "Confidence", "Price", "Reasons"]

        def _style_scr(val):
            c = {"BUY": "color:#00e676;font-weight:bold",
                 "SELL": "color:#ff5252;font-weight:bold"}
            return c.get(val, "")

        st.dataframe(
            scr_df[["Symbol", "Signal", "Strategy", "Confidence", "Price"]].style
            .map(_style_scr, subset=["Signal"]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")
        chosen = st.selectbox(
            "Drill into a stock",
            [r["symbol"].replace(".NS", "") for r in results],
        )
        if st.button(f"📊 Analyze {chosen}"):
            with st.spinner(f"Analysing {chosen}…"):
                res = orch.analyze(chosen)
                st.session_state["analysis_result"] = res
            st.info(f"✅ {chosen} loaded — switch to the **Analysis** tab to view.")


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and informational purposes only. "
    "It does NOT constitute financial advice. Past performance is not indicative of future results. "
    "Always do your own research before trading.  |  Data via Yahoo Finance (free tier)"
)
