"""
Chat UI for the finance analyst agent. Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # let it import server/ and client/ as top-level packages

import streamlit as st
from client.ollama_bridge import run_agent_sync, DEFAULT_MODEL

st.set_page_config(page_title="FinMCP — Autonomous Data Analyst", page_icon="📈", layout="centered")

st.title("📈 FinMCP — Autonomous Finance Analyst")
st.caption(
    "An MCP-powered agent for stock analysis, running on a local Ollama model. "
    "Every numeric answer is grounded in a real tool call — nothing is guessed."
)

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    ollama_host = st.text_input("Ollama host (optional)", value="", placeholder="http://localhost:11434")
    think = st.checkbox(
        "Enable thinking mode", value=False,
        help="Qwen3 defaults to an extended reasoning trace before every response. "
             "Off is much faster, especially on CPU-only machines — on by default in "
             "Ollama, off by default here."
    )
    st.markdown("---")
    st.markdown(
        "**Try asking:**\n"
        "- How has AAPL performed over the last 6 months?\n"
        "- Compare MSFT and GOOGL over the last year\n"
        "- Has TSLA had any unusual price spikes recently?\n"
        "- What's NVDA's 50-day moving average?\n"
        "- Plot AMZN's price chart for the last year"
    )

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("chart_path"):
            st.image(turn["chart_path"])
        if turn.get("tool_calls"):
            with st.expander(f"🔧 {len(turn['tool_calls'])} tool call(s)"):
                for tc in turn["tool_calls"]:
                    status = f"❌ {tc.error}" if tc.error else "✅"
                    st.markdown(f"**{tc.name}**`({tc.arguments})` — {tc.elapsed_s}s {status}")
                    st.code(tc.result_preview, language="json")

query = st.chat_input("Ask a question about a stock...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Planning and calling tools..."):
            try:
                result = run_agent_sync(
                    query,
                    model=model or DEFAULT_MODEL,
                    ollama_host=ollama_host or None,
                    think=think,
                )
                answer = result.final_answer
                chart_path = next(
                    (tc.result_preview for tc in result.tool_calls if tc.name == "plot_price_chart"),
                    None,
                )
                # extract the actual path from the tool's JSON text, not just show raw text
                chart_file = None
                if chart_path and '"chart_path"' in chart_path:
                    import json as _json
                    try:
                        chart_file = _json.loads(chart_path).get("chart_path")
                    except Exception:
                        chart_file = None

                st.markdown(answer)
                if chart_file:
                    st.image(chart_file)

                with st.expander(f"🔧 {len(result.tool_calls)} tool call(s) · {result.total_elapsed_s}s total"):
                    for tc in result.tool_calls:
                        status = f"❌ {tc.error}" if tc.error else "✅"
                        st.markdown(f"**{tc.name}**`({tc.arguments})` — {tc.elapsed_s}s {status}")
                        st.code(tc.result_preview, language="json")

                st.session_state.history.append({
                    "role": "assistant", "content": answer,
                    "chart_path": chart_file, "tool_calls": result.tool_calls,
                })
            except Exception as e:
                st.error(
                    f"{e}\n\n"
                    "Common causes: Ollama isn't running (`ollama serve`), the model "
                    "isn't pulled (`ollama pull qwen3:8b`), or there's no internet "
                    "access for the market-data fetch."
                )
