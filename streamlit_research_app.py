from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

RENDER_MERMAID_GRAPH = False  # Set to True to render Mermaid graphs in the Streamlit app (requires JS support)

DEFAULT_API_URL = os.getenv("ENTITY_LINKING_API_URL", "http://localhost")
DEFAULT_ENDPOINT = os.getenv("ENTITY_LINKING_RESEARCH_ENDPOINT", "/agent/research")
DEFAULT_TIMEOUT = int(os.getenv("ENTITY_LINKING_RESEARCH_TIMEOUT", "600"))


# ── Helpers ──────────────────────────────────────────────────────────────────


def build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def try_parse_json(value: str) -> Optional[Any]:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def escape_md(text: str) -> str:
    """Escape colons that form :word: emoji shortcodes without breaking prefixed URIs."""
    import re
    return re.sub(r":([a-zA-Z0-9_+-]+):", r"&#58;\1&#58;", text)


def render_mermaid(mermaid_src: str) -> None:
    """Render a Mermaid diagram. Uses textContent + mermaid.run() to avoid
    HTML-escaping issues that break Mermaid syntax characters like > and :.
    Auto-resizes the iframe to fit the rendered SVG."""
    encoded = json.dumps(mermaid_src)
    markup = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  .mermaid {{ padding:8px; }}
  .mermaid svg {{ display:block; max-width:100%; height:auto; }}
</style>
</head>
<body>
<pre id="graph" class="mermaid"></pre>
<script>
  mermaid.initialize({{ startOnLoad:false, theme:'neutral', securityLevel:'loose' }});
  document.getElementById('graph').textContent = {encoded};
  mermaid.run({{ nodes: [document.getElementById('graph')] }}).then(function() {{
    setTimeout(function() {{
      var svg = document.querySelector('svg');
      if (svg) {{
        var h = svg.getBoundingClientRect().height + 20;
        document.body.style.height = h + 'px';
        window.parent.postMessage({{type:'streamlit:setFrameHeight', height: h}}, '*');
      }}
    }}, 150);
  }});
</script>
</body>
</html>"""
    components.html(markup, height=800, scrolling=False)


def render_value(label: str, value: Any, key: str) -> None:
    if value is None or value == "":
        st.caption("No value returned.")
        return
    if isinstance(value, (dict, list)):
        st.json(value)
        return
    text = str(value)
    parsed = try_parse_json(text)
    if parsed is not None:
        st.json(parsed)
        return
    height = min(520, max(120, 80 + len(text) // 4))
    st.text_area(label, text, height=height, key=key, label_visibility="collapsed")


# ── API calls ────────────────────────────────────────────────────────────────


def call_research_endpoint(
    base_url: str,
    endpoint: str,
    query: str,
    messages: list,
    timeout: int,
    bearer_token: str,
    verify_tls: bool,
) -> Tuple[int, Dict[str, Any], float, str]:
    url = build_url(base_url, endpoint)
    headers = {"Content-Type": "application/json"}
    if bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"

    body = {"query": query}
    if messages:
        body["messages"] = messages

    started_at = time.perf_counter()
    response = requests.post(url, json=body, headers=headers, timeout=timeout, verify=verify_tls)
    elapsed = time.perf_counter() - started_at

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_text": response.text}

    return response.status_code, payload, elapsed, url


def call_research_stream(
    base_url: str,
    endpoint: str,
    query: str,
    messages: list,
    timeout: int,
    bearer_token: str,
    verify_tls: bool,
):
    """Call the streaming research endpoint. Yields parsed SSE events."""
    stream_endpoint = endpoint.rstrip("/") + "/stream"
    url = build_url(base_url, stream_endpoint)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"

    body = {"query": query}
    if messages:
        body["messages"] = messages

    with requests.post(
        url, json=body, headers=headers, timeout=timeout, verify=verify_tls, stream=True
    ) as response:
        if response.status_code != 200:
            yield {"event": "error", "data": {"detail": f"HTTP {response.status_code}: {response.text}"}}
            return
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    yield json.loads(line[6:])
                except json.JSONDecodeError:
                    pass


# ── Data helpers ─────────────────────────────────────────────────────────────


def payload_answer(payload: Dict[str, Any]) -> str:
    return str(payload.get("answer") or "")


def build_chat_messages(turns: list, query: str) -> list:
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn["query"]})
        answer = payload_answer(turn.get("payload", {}))
        if answer:
            messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": query})
    return messages


def tool_call_name(tool_call: Dict[str, Any]) -> str:
    if tool_call.get("name"):
        return str(tool_call["name"])
    function = tool_call.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return "tool_call"


def tool_call_args(tool_call: Dict[str, Any]) -> Any:
    if "args" in tool_call:
        return tool_call["args"]
    function = tool_call.get("function")
    if isinstance(function, dict) and "arguments" in function:
        arguments = function["arguments"]
        if isinstance(arguments, str):
            parsed = try_parse_json(arguments)
            return parsed if parsed is not None else arguments
        return arguments
    return None


# ── Rendering components ─────────────────────────────────────────────────────


def render_plan_progress(
    plan_steps: list, step_results: list, active_step: int = -1
) -> None:
    """Render plan steps with execution status. active_step marks the running step."""
    if not plan_steps:
        st.caption("No plan data available.")
        return

    result_by_index = {sr.get("step_index", -1): sr for sr in step_results}
    status_icons = {
        "completed": "\u2705",
        "failed": "\u274c",
        "skipped": "\u23ed\ufe0f",
        "timed_out": "\u23f3",
    }

    for step in plan_steps:
        idx = step.get("index", 0)
        goal = escape_md(step.get("goal", ""))
        tools = ", ".join(step.get("tools_needed", [])) or None
        result = result_by_index.get(idx)

        if result:
            status = result.get("status", "unknown")
            elapsed_s = result.get("elapsed_s", 0)
            tc_count = result.get("tool_calls_count", 0)
            icon = status_icons.get(status, "\u2753")
            st.markdown(
                f"{icon} **Step {idx}** \u2014 {goal}  \n"
                f"<small style='color:gray'>{status} \u00b7 {elapsed_s:.1f}s"
                f" \u00b7 {tc_count} calls</small>",
                unsafe_allow_html=True,
            )
            if result.get("result_summary"):
                st.markdown(escape_md(result["result_summary"]), unsafe_allow_html=True)
        elif idx == active_step:
            st.markdown(
                f"\u25b6\ufe0f **Step {idx}** \u2014 {goal}", unsafe_allow_html=True
            )
            if tools:
                st.caption(f"Tools: {escape_md(tools)}")
        else:
            st.markdown(
                f"\u2b1c **Step {idx}** \u2014 {goal}", unsafe_allow_html=True
            )
            if tools:
                st.caption(f"Tools: {escape_md(tools)}")


def render_details_tabs(turn: Dict[str, Any], turn_index: int) -> None:
    """Render detailed response data in tabs. Only tabs with data are shown."""
    payload = turn["payload"]
    key_prefix = f"turn-{turn_index}"

    raw_response = payload.get("raw_response") or {}
    plan_steps = raw_response.get("plan") or []
    step_results_data = raw_response.get("step_results") or []
    messages_data = payload.get("messages")
    tool_calls_data = payload.get("tool_calls")
    tool_results_data = payload.get("tool_results") or payload.get("sparql_results")
    trace = payload.get("trace")
    diagram = turn.get("diagram")

    tab_names: list[str] = []
    tab_keys: list[str] = []

    if plan_steps or step_results_data:
        tab_names.append("Plan & Progress")
        tab_keys.append("plan")
    if diagram and RENDER_MERMAID_GRAPH:
        tab_names.append("Knowledge Graph")
        tab_keys.append("graph")
    if messages_data:
        tab_names.append("Messages")
        tab_keys.append("messages")
    if tool_calls_data:
        tab_names.append("Tool Calls")
        tab_keys.append("tool_calls")
    if tool_results_data:
        tab_names.append("Tool Results")
        tab_keys.append("tool_results")
    if trace:
        tab_names.append("Trace")
        tab_keys.append("trace")
    tab_names.append("Raw JSON")
    tab_keys.append("raw_json")

    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_keys, tabs))

    if "plan" in tab_map:
        with tab_map["plan"]:
            render_plan_progress(plan_steps, step_results_data)

    if "graph" in tab_map:
        with tab_map["graph"]:
            if RENDER_MERMAID_GRAPH:
                render_mermaid(diagram)

    if "messages" in tab_map:
        with tab_map["messages"]:
            for msg in messages_data:
                index = msg.get("index", "?")
                role = msg.get("role", "message")
                msg_type = msg.get("type", "unknown")
                tc_count = len(msg.get("tool_calls") or [])
                suffix = f" ({tc_count} tool calls)" if tc_count else ""
                with st.container(border=True):
                    st.markdown(f"**{index}. {role}** ({msg_type}){suffix}")
                    content = msg.get("content")
                    if content:
                        render_value(
                            "content", content, key=f"{key_prefix}-msg-{index}"
                        )

    if "tool_calls" in tab_map:
        with tab_map["tool_calls"]:
            for i, tc in enumerate(tool_calls_data, 1):
                name = tool_call_name(tc)
                msg_idx = tc.get("message_index", "?")
                with st.container(border=True):
                    st.markdown(f"**{i}. {name}** (message {msg_idx})")
                    args = tool_call_args(tc)
                    if args is not None:
                        st.json(
                            args if isinstance(args, (dict, list)) else {"value": args}
                        )

    if "tool_results" in tab_map:
        with tab_map["tool_results"]:
            for i, result in enumerate(tool_results_data, 1):
                tool = result.get("tool", "unknown")
                call_id = result.get("tool_call_id", "")
                with st.container(border=True):
                    st.markdown(f"**{i}. {tool}** ({call_id})")
                    content = result.get("content", result.get("result"))
                    if content:
                        render_value(
                            "result", content, key=f"{key_prefix}-tr-{i}"
                        )

    if "trace" in tab_map:
        with tab_map["trace"]:
            render_value("trace", trace, key=f"{key_prefix}-trace")

    with tab_map["raw_json"]:
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=f"response-{turn_index}.json",
            mime="application/json",
            key=f"download-{turn_index}",
        )
        st.json(payload)


def render_turn(turn: Dict[str, Any], turn_index: int) -> None:
    """Render a completed conversation turn (history)."""
    payload = turn["payload"]
    status_code = turn["status_code"]
    elapsed = turn["elapsed"]

    with st.chat_message("user"):
        st.markdown(turn["query"])

    with st.chat_message("assistant"):
        if status_code == 0 or status_code >= 400:
            st.error(f"HTTP {status_code}")
            render_value("Error", payload, key=f"turn-{turn_index}-error")
            return

        answer = payload_answer(payload) or payload.get("detail") or ""
        st.markdown(answer if answer else "_No answer returned._")
        st.caption(f"{elapsed:.1f}s")

        with st.expander("Details", expanded=False):
            render_details_tabs(turn, turn_index)


# ── Page layout ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Research Mode", layout="wide")
st.title("Research Mode")

if "turns" not in st.session_state:
    st.session_state.turns = []
if "diagram" not in st.session_state:
    st.session_state.diagram = None

# ── Sidebar ──

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
    endpoint = st.text_input("Research endpoint", value=DEFAULT_ENDPOINT)
    timeout = st.number_input(
        "Timeout (s)", min_value=5, max_value=900, value=DEFAULT_TIMEOUT, step=5
    )
    bearer_token = st.text_input("Bearer token", value="", type="password")
    verify_tls = st.checkbox("Verify TLS", value=True)
    stream_progress = st.checkbox("Stream progress", value=True)

    st.divider()

    if st.button("Check backend"):
        try:
            health = requests.get(
                api_url.rstrip("/") + "/", timeout=10, verify=verify_tls
            )
            if health.ok:
                st.success(f"HTTP {health.status_code}")
                try:
                    st.json(health.json())
                except ValueError:
                    st.text(health.text)
            else:
                st.error(f"HTTP {health.status_code}")
        except requests.RequestException as exc:
            st.error(str(exc))

    if st.button("Clear chat"):
        st.session_state.turns = []
        st.session_state.diagram = None
        st.rerun()

# ── Chat history ──

for idx, stored_turn in enumerate(st.session_state.turns, start=1):
    render_turn(stored_turn, idx)

# ── Chat input ──

prompt = st.chat_input("Ask a research question")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = build_chat_messages(st.session_state.turns, prompt)

        if stream_progress:
            # ── Streaming mode ──
            # Status bar at top, tabs section below that updates as data arrives.
            # 1. graph  → Knowledge Graph tab appears
            # 2. plan   → Plan & Progress tab appears (becomes active)
            # 3. steps  → Plan & Progress updates in-place
            # 4. done   → final answer shown, tabs persist
            started_at = time.perf_counter()
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            tabs_container = st.empty()

            status_placeholder.info("Connecting...")
            plan_data: list = []
            step_results_data: list = []
            final_answer = ""
            had_error = False
            active_step_idx = -1
            mermaid_diagram: Optional[str] = None

            def render_live_tabs() -> None:
                """Re-render the bottom tabs with current state."""
                with tabs_container.container():
                    tab_names: list[str] = []
                    tab_keys: list[str] = []
                    if mermaid_diagram and not plan_data and RENDER_MERMAID_GRAPH:
                        # Before plan: graph tab only (shown first)
                        tab_names.append("Knowledge Graph")
                        tab_keys.append("graph")
                    if plan_data:
                        # Once plan exists: plan first, graph second
                        tab_names.append("Plan & Progress")
                        tab_keys.append("plan")
                        if mermaid_diagram and RENDER_MERMAID_GRAPH:
                            tab_names.append("Knowledge Graph")
                            tab_keys.append("graph")
                    if not tab_names:
                        return
                    tabs = st.tabs(tab_names)
                    tab_map = dict(zip(tab_keys, tabs))
                    if "plan" in tab_map:
                        with tab_map["plan"]:
                            render_plan_progress(
                                plan_data, step_results_data, active_step_idx
                            )
                    if "graph" in tab_map:
                        with tab_map["graph"]:
                            if RENDER_MERMAID_GRAPH:
                                render_mermaid(mermaid_diagram)

            try:
                for event in call_research_stream(
                    api_url,
                    endpoint,
                    prompt,
                    messages,
                    int(timeout),
                    bearer_token,
                    verify_tls,
                ):
                    ev_type = event.get("event", "")
                    ev_data = event.get("data", {})

                    if ev_type == "graph":
                        mermaid_src = ev_data.get("mermaid", "")
                        if mermaid_src:
                            mermaid_diagram = mermaid_src
                            st.session_state.diagram = mermaid_src
                        status_placeholder.info(
                            "Graph built. Retrieving documentation..."
                        )
                        render_live_tabs()

                    elif ev_type == "retrieve":
                        docs_len = ev_data.get("docs_length", 0)
                        status_placeholder.info(
                            f"Retrieved {docs_len} chars. Planning..."
                        )

                    elif ev_type == "plan":
                        plan_data = ev_data.get("plan", [])
                        active_step_idx = 0
                        status_placeholder.info(
                            f"Plan: {len(plan_data)} step(s). Executing..."
                        )
                        render_live_tabs()

                    elif ev_type == "step_done":
                        step_result = ev_data.get("step_result")
                        if step_result:
                            step_results_data.append(step_result)
                        current_idx = ev_data.get("current_step_index", 0)
                        total = len(plan_data)
                        if current_idx < total:
                            active_step_idx = current_idx
                            status_placeholder.info(
                                f"Step {current_idx}/{total} done. Continuing..."
                            )
                        else:
                            active_step_idx = -1
                            status_placeholder.info(
                                "All steps done. Validating..."
                            )
                        render_live_tabs()

                    elif ev_type == "replan":
                        plan_data = ev_data.get("plan", [])
                        step_results_data = []
                        active_step_idx = 0
                        status_placeholder.warning(
                            f"Replanned: {len(plan_data)} step(s)"
                        )
                        render_live_tabs()

                    elif ev_type == "validate":
                        final_answer = ev_data.get("answer", "")
                        status_placeholder.success("Complete")

                    elif ev_type == "done":
                        status_placeholder.empty()

                    elif ev_type == "error":
                        status_placeholder.error(
                            ev_data.get("detail", "Unknown error")
                        )
                        had_error = True

            except requests.RequestException as exc:
                status_placeholder.error(str(exc))
                had_error = True

            elapsed = time.perf_counter() - started_at
            status_placeholder.empty()

            # Build turn
            payload = {
                "answer": final_answer,
                "raw_response": {
                    "plan": plan_data,
                    "step_results": step_results_data,
                },
            }
            turn = {
                "query": prompt,
                "payload": payload,
                "status_code": 200 if not had_error else 500,
                "elapsed": elapsed,
                "url": build_url(api_url, endpoint + "/stream"),
                "messages_sent": messages,
                "diagram": mermaid_diagram,
            }

            # Show final answer above tabs
            with answer_placeholder.container():
                if final_answer:
                    st.markdown(final_answer)
                else:
                    st.markdown("_No answer returned._")
                st.caption(f"{elapsed:.1f}s")

            # Update tabs to final state (no active step, diagram stays)
            active_step_idx = -1
            render_live_tabs()

        else:
            # ── Regular mode: single POST ──
            with st.spinner("Waiting for response..."):
                try:
                    status_code, payload, elapsed, url = call_research_endpoint(
                        api_url,
                        endpoint,
                        prompt,
                        messages,
                        int(timeout),
                        bearer_token,
                        verify_tls,
                    )
                    turn = {
                        "query": prompt,
                        "payload": payload,
                        "status_code": status_code,
                        "elapsed": elapsed,
                        "url": url,
                        "messages_sent": messages,
                    }
                except requests.RequestException as exc:
                    turn = {
                        "query": prompt,
                        "payload": {"detail": str(exc)},
                        "status_code": 0,
                        "elapsed": 0.0,
                        "url": build_url(api_url, endpoint),
                        "messages_sent": messages,
                    }

            status_code = turn["status_code"]
            if status_code == 0 or status_code >= 400:
                st.error(f"HTTP {status_code}")
                render_value("Error", turn["payload"], key="current-error")
            else:
                answer = payload_answer(turn["payload"])
                st.markdown(answer if answer else "_No answer returned._")
                st.caption(f"{turn['elapsed']:.1f}s")
                with st.expander("Details", expanded=False):
                    render_details_tabs(turn, len(st.session_state.turns) + 1)

    st.session_state.turns.append(turn)
