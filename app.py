import re
import time

import streamlit as st

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Research System")
    st.caption("Four cooperating AI agents turn a topic into a reviewed report.")

    st.markdown("### Pipeline")
    st.markdown(
        """
1. **🌐 Search Agent** – finds recent, reliable sources  
2. **📖 Reader Agent** – scrapes the best URL for depth  
3. **✍️ Writer Chain** – drafts a structured report  
4. **🧐 Critic Chain** – scores and critiques the report
"""
    )

    st.markdown("### Try an example")
    examples = [
        "Impact of AI on healthcare in 2026",
        "Latest developments in solid-state batteries",
        "State of quantum computing",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["topic_input"] = ex

    st.divider()
    if st.button("🗑️ Clear results", use_container_width=True):
        st.session_state.pop("state", None)
        st.session_state.pop("timings", None)
        st.session_state.pop("last_topic", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Header + input
# ---------------------------------------------------------------------------
st.title("Multi-Agent Research System")
st.write("Enter a topic and let the agents search, read, write and review for you.")

col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_input:
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Impact of AI on healthcare in 2026",
    )
with col_btn:
    run_clicked = st.button("🚀 Run", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Pipeline (same logic as pipeline.py, but reports progress to the UI)
# ---------------------------------------------------------------------------
def run_pipeline(topic: str):
    state, timings = {}, {}

    # Step 1 - Search
    with st.status("Step 1 · Search agent is working...", expanded=False) as status:
        t = time.time()
        search_agent = build_search_agent()
        result = search_agent.invoke(
            {
                "messages": [
                    ("user", f"Find recent, reliable and detailed information about: {topic}")
                ]
            }
        )
        state["search_results"] = result["messages"][-1].content
        timings["Search"] = time.time() - t
        status.update(label="Step 1 · Search complete ✅", state="complete")

    # Step 2 - Reader
    with st.status("Step 2 · Reader agent is scraping top resources...", expanded=False) as status:
        t = time.time()
        reader_agent = build_reader_agent()
        result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = result["messages"][-1].content
        timings["Reader"] = time.time() - t
        status.update(label="Step 2 · Scraping complete ✅", state="complete")

    # Step 3 - Writer
    with st.status("Step 3 · Writer is drafting the report...", expanded=False) as status:
        t = time.time()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke(
            {"topic": topic, "research": research_combined}
        )
        timings["Writer"] = time.time() - t
        status.update(label="Step 3 · Report drafted ✅", state="complete")

    # Step 4 - Critic
    with st.status("Step 4 · Critic is reviewing the report...", expanded=False) as status:
        t = time.time()
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        timings["Critic"] = time.time() - t
        status.update(label="Step 4 · Review complete ✅", state="complete")

    return state, timings


def extract_score(feedback: str):
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Run on click
# ---------------------------------------------------------------------------
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        try:
            state, timings = run_pipeline(topic.strip())
            st.session_state["state"] = state
            st.session_state["timings"] = timings
            st.session_state["last_topic"] = topic.strip()
        except Exception as e:
            st.error(f"Something went wrong while running the pipeline: {e}")

# ---------------------------------------------------------------------------
# Display results (persisted in session_state so they survive reruns)
# ---------------------------------------------------------------------------
if "state" in st.session_state:
    state = st.session_state["state"]
    timings = st.session_state.get("timings", {})
    last_topic = st.session_state.get("last_topic", "")

    st.divider()
    st.subheader(f"Results for: {last_topic}")

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    score = extract_score(state["feedback"])
    m1.metric("Critic score", f"{score}/10" if score else "—")
    m2.metric("Report length", f"{len(state['report'].split())} words")
    m3.metric("Total time", f"{sum(timings.values()):.1f}s")

    tab_report, tab_critic, tab_search, tab_scrape = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🌐 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critic:
        st.markdown(state["feedback"])

    with tab_search:
        st.markdown(state["search_results"])

    with tab_scrape:
        st.markdown(state["scraped_content"])

    if timings:
        with st.expander("⏱️ Step timings"):
            for name, secs in timings.items():
                st.write(f"**{name}**   .venv\Scripts\python.exe -m streamlit run app.py
                         : {secs:.1f}s")