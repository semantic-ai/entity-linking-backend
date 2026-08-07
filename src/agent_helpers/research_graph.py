"""
LangGraph-based research agent with Plan-then-Execute architecture.

Graph: Retrieve → Plan → Execute (per-step) → Monitor → Replan/Validate → END

The monitor is pure Python (no LLM calls), checking for stuck/timeout conditions
and whether results warrant replanning.  Replanning uses a single LLM call to
revise the remaining plan steps based on what was learned so far.
"""

import concurrent.futures
import json
import time
from typing import Annotated, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from helpers import logger
from src.agent_helpers.logging_callbacks import AgentStepLogger
from src.agent_helpers.prompts import REPLAN_PROMPT, RETRIEVE_SYSTEM_PROMPT, PLANNER_PROMPT, EXECUTE_STEP_PROMPT, VALIDATE_PROMPT

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    """A single step in the research plan."""
    index: int = Field(description="Step number (0-based)")
    goal: str = Field(description="What this step accomplishes")
    tools_needed: List[str] = Field(default_factory=list, description="Which tools to use")
    success_criteria: str = Field(default="", description="How to know the step succeeded")
    fallback: str = Field(default="", description="Alternative approach if this fails")
    timeout_s: int = Field(default=45, description="Max seconds for this step")


class StepResult(BaseModel):
    """Tracking data for a completed plan step."""
    step_index: int
    status: str  # "completed" | "failed" | "skipped" | "timed_out"
    tool_calls_count: int = 0
    elapsed_s: float = 0.0
    result_summary: str = ""


class ResearchGraphState(TypedDict):
    """State that flows through the research graph."""
    # Input
    query: str
    messages: Annotated[list, add_messages]

    # Retrieve phase
    retrieved_docs: str

    # Plan phase
    plan: list  # list of PlanStep dicts
    current_step_index: int
    replan_count: int  # how many times we've replanned

    # Monitor state
    step_start_time: float

    # Results
    step_results: list  # list of StepResult dicts
    final_answer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ai_content(msgs: list) -> str:
    """Return the last non-empty AI message content from a message list."""
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                return content
    return ""


def _count_tool_messages(msgs: list) -> int:
    return sum(1 for msg in msgs if isinstance(msg, ToolMessage))


def _step_looks_unsuccessful(step_result: dict) -> bool:
    """Heuristic: did a step fail to produce useful output?"""
    status = step_result.get("status", "")
    if status in ("failed", "timed_out", "skipped"):
        return True
    summary = step_result.get("result_summary", "").lower()
    failure_signals = [
        "no results", "0 results", "zero results",
        "returned empty", "could not find", "did not find",
        "no matching", "no data found", "returned no",
    ]
    return any(signal in summary for signal in failure_signals)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class ResearchGraphBuilder:
    """Builds and compiles the research LangGraph.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM to use for all phases.
    tools : list[StructuredTool]
        All available LangChain tools (from MCP).
    retrieve_tool_names : list[str]
        Tool names available during the retrieve phase.
    execute_tool_names : list[str] | None
        Tool names available during execution. None = all tools.
    step_timeout_s : int
        Default per-step timeout in seconds.
    max_step_tool_calls : int
        Max tool calls within a single plan step.
    max_replans : int
        Max number of times the plan can be revised.
    """

    def __init__(
        self,
        llm,
        tools: list,
        retrieve_tool_names: Optional[List[str]] = None,
        execute_tool_names: Optional[List[str]] = None,
        step_timeout_s: int = 45,
        max_step_tool_calls: int = 10,
        max_replans: int = 2,
    ):
        self.llm = llm
        self.all_tools = {t.name: t for t in tools}
        self.retrieve_tool_names = retrieve_tool_names or ["search_sparql_docs"]
        self.execute_tool_names = execute_tool_names  # None = all
        self.step_timeout_s = step_timeout_s
        self.max_step_tool_calls = max_step_tool_calls
        self.max_replans = max_replans

    def _get_tools(self, names: Optional[List[str]] = None) -> list:
        """Get tools by name, falling back to all tools."""
        if names is None:
            return list(self.all_tools.values())
        return [self.all_tools[n] for n in names if n in self.all_tools]

    # ------------------------------------------------------------------
    # Node: RETRIEVE
    # ------------------------------------------------------------------
    def retrieve_node(self, state: ResearchGraphState) -> dict:
        """Retrieve relevant documentation using search tools."""
        logger.info("[GRAPH:RETRIEVE] Starting retrieval phase...")

        retrieve_tools = self._get_tools(self.retrieve_tool_names)
        if not retrieve_tools:
            logger.warning("[GRAPH:RETRIEVE] No retrieve tools available, skipping.")
            return {"retrieved_docs": ""}

        retrieve_agent = create_react_agent(
            self.llm,
            retrieve_tools,
            prompt=RETRIEVE_SYSTEM_PROMPT,
        )

        # Use conversation history so follow-up questions have context
        input_messages = state.get("messages", [HumanMessage(content=state["query"])])

        result = retrieve_agent.invoke(
            {"messages": input_messages},
            {"recursion_limit": 10},
        )

        msgs = result.get("messages", [])
        docs_text = _extract_ai_content(msgs)

        # Also collect raw tool outputs as supplementary context
        tool_outputs = []
        for msg in msgs:
            if isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    tool_outputs.append(content)

        if tool_outputs:
            docs_text += "\n\n## Raw Tool Outputs\n" + "\n---\n".join(tool_outputs)

        logger.info(f"[GRAPH:RETRIEVE] Retrieved {len(docs_text)} chars of documentation.")
        return {"retrieved_docs": docs_text}

    # ------------------------------------------------------------------
    # Node: PLAN
    # ------------------------------------------------------------------
    def plan_node(self, state: ResearchGraphState) -> dict:
        """Generate a structured plan based on query + retrieved docs."""
        logger.info("[GRAPH:PLAN] Generating execution plan...")

        # Build conversation history summary for the planner
        conversation_history = ""
        raw_messages = state.get("messages", [])
        if len(raw_messages) > 1:
            for msg in raw_messages[:-1]:  # exclude last (current query)
                role = getattr(msg, "type", "") if hasattr(msg, "type") else msg.get("role", "")
                content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")
                if role and content:
                    conversation_history += f"**{role}**: {content}\n"

        prompt = PLANNER_PROMPT.format(
            retrieved_docs=state.get("retrieved_docs", "(no documentation retrieved)"),
            conversation_history=conversation_history or "(no prior conversation)",
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["query"]),
        ])

        plan_text = response.content if isinstance(response.content, str) else str(response.content)
        plan_steps = self._parse_plan(plan_text)

        logger.info(f"[GRAPH:PLAN] Created plan with {len(plan_steps)} steps:")
        for step in plan_steps:
            logger.info(f"  Step {step['index']}: {step['goal']}")

        return {
            "plan": plan_steps,
            "current_step_index": 0,
            "replan_count": 0,
            "step_start_time": time.time(),
            "step_results": [],
        }

    # ------------------------------------------------------------------
    # Node: REPLAN
    # ------------------------------------------------------------------
    def replan_node(self, state: ResearchGraphState) -> dict:
        """Revise remaining plan steps based on what was learned so far."""
        replan_count = state.get("replan_count", 0) + 1
        logger.info(f"[GRAPH:REPLAN] Replanning (attempt {replan_count})...")

        # Build summaries of completed steps
        step_summaries = ""
        for sr in state.get("step_results", []):
            step_summaries += (
                f"\n### Step {sr['step_index']} (status: {sr['status']})\n"
                f"{sr.get('result_summary', '(no summary)')}\n"
            )

        # Build remaining steps that haven't executed yet
        plan = state.get("plan", [])
        current_idx = state.get("current_step_index", 0)
        remaining = [s for s in plan if s.get("index", 0) >= current_idx]
        remaining_text = json.dumps(remaining, indent=2) if remaining else "(none)"

        prompt = REPLAN_PROMPT.format(
            query=state["query"],
            retrieved_docs=state.get("retrieved_docs", "(none)"),
            step_summaries=step_summaries or "(none)",
            remaining_steps=remaining_text,
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["query"]),
        ])

        plan_text = response.content if isinstance(response.content, str) else str(response.content)
        new_steps = self._parse_plan(plan_text)

        # Re-index from current position
        for i, step in enumerate(new_steps):
            step["index"] = current_idx + i

        # Merge: keep completed steps + new steps
        completed_plan = [s for s in plan if s.get("index", 0) < current_idx]
        merged_plan = completed_plan + new_steps

        logger.info(f"[GRAPH:REPLAN] Revised plan: {len(new_steps)} new steps (total {len(merged_plan)})")
        for step in new_steps:
            logger.info(f"  Step {step['index']}: {step['goal']}")

        return {
            "plan": merged_plan,
            "replan_count": replan_count,
            "step_start_time": time.time(),
        }

    def _parse_plan(self, plan_text: str) -> list:
        """Parse LLM output into a list of PlanStep dicts. Graceful fallback on failure."""
        text = plan_text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            raw = json.loads(text)
            if isinstance(raw, list):
                steps = []
                for i, item in enumerate(raw):
                    step = PlanStep(
                        index=item.get("index", i),
                        goal=item.get("goal", f"Step {i}"),
                        tools_needed=item.get("tools_needed", []),
                        success_criteria=item.get("success_criteria", ""),
                        fallback=item.get("fallback", ""),
                        timeout_s=item.get("timeout_s", self.step_timeout_s),
                    )
                    # Filter tools to only those that actually exist
                    step.tools_needed = [t for t in step.tools_needed if t in self.all_tools]
                    steps.append(step.model_dump())
                return steps
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"[GRAPH:PLAN] Failed to parse plan JSON: {e}. Using fallback plan.")

        # Fallback: single step that does everything
        fallback = PlanStep(
            index=0,
            goal="Research and answer the user's question using available tools",
            tools_needed=list(self.all_tools.keys()),
            success_criteria="A complete answer to the question",
            fallback="Try simpler queries or different search terms",
            timeout_s=self.step_timeout_s * 2,
        )
        return [fallback.model_dump()]

    # ------------------------------------------------------------------
    # Node: EXECUTE
    # ------------------------------------------------------------------
    def execute_node(self, state: ResearchGraphState) -> dict:
        """Execute the current plan step using a ReAct sub-agent."""
        step_index = state["current_step_index"]
        plan = state["plan"]

        if step_index >= len(plan):
            return {}

        step = plan[step_index]
        logger.info(f"[GRAPH:EXECUTE] Running step {step_index}: {step['goal']}")

        # Determine tools for this step
        step_tool_names = step.get("tools_needed") or None
        if self.execute_tool_names is not None:
            if step_tool_names:
                step_tool_names = [t for t in step_tool_names if t in self.execute_tool_names]
            else:
                step_tool_names = self.execute_tool_names

        execute_tools = self._get_tools(step_tool_names)
        if not execute_tools:
            execute_tools = self._get_tools()

        # Build context from previous step results
        previous_context = ""
        for sr in state.get("step_results", []):
            previous_context += f"\n### Step {sr['step_index']} ({sr['status']})\n{sr['result_summary']}\n"
        if not previous_context:
            previous_context = "(This is the first step)"

        # Build conversation context for follow-up awareness
        conversation_context = ""
        raw_messages = state.get("messages", [])
        if len(raw_messages) > 1:
            for msg in raw_messages[:-1]:
                role = getattr(msg, "type", "") if hasattr(msg, "type") else msg.get("role", "")
                content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")
                if role and content:
                    conversation_context += f"**{role}**: {content}\n"

        step_prompt = EXECUTE_STEP_PROMPT.format(
            step_index=step_index,
            step_goal=step["goal"],
            success_criteria=step.get("success_criteria", "Complete the goal"),
            fallback=step.get("fallback", "Try a different approach"),
            previous_context=previous_context,
            retrieved_docs=state.get("retrieved_docs", "(no documentation available)"),
            conversation_history=conversation_context or "(no prior conversation)",
        )

        step_agent = create_react_agent(
            self.llm,
            execute_tools,
            prompt=step_prompt,
        )

        # Enforce per-step timeout — clamp between configured default and 120s
        # The LLM sometimes generates low timeout_s values (e.g. 20s) that expire
        # before a tool call even completes. Use step_timeout_s as the floor.
        step_timeout = max(self.step_timeout_s, min(step.get("timeout_s", self.step_timeout_s), 120))

        # NOTE: Do NOT use `with executor:` — the context manager calls
        # shutdown(wait=True) which blocks until the thread finishes,
        # completely defeating the timeout.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            step_logger = AgentStepLogger()
            future = executor.submit(
                step_agent.invoke,
                {"messages": [HumanMessage(content=state["query"])]},
                {"recursion_limit": self.max_step_tool_calls, "callbacks": [step_logger]},
            )
            result = future.result(timeout=step_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            logger.warning(
                f"[GRAPH:EXECUTE] Step {step_index} timed out after {step_timeout}s"
            )
            step_result = StepResult(
                step_index=step_index,
                status="timed_out",
                elapsed_s=time.time() - state.get("step_start_time", time.time()),
                result_summary=f"Step timed out after {step_timeout}s. The query may be too complex or the endpoint unresponsive.",
            )
            return {
                "step_results": state.get("step_results", []) + [step_result.model_dump()],
                "current_step_index": step_index + 1,
                "step_start_time": time.time(),
            }
        except Exception as e:
            executor.shutdown(wait=False)
            logger.error(f"[GRAPH:EXECUTE] Step {step_index} failed: {e}")
            step_result = StepResult(
                step_index=step_index,
                status="failed",
                elapsed_s=time.time() - state.get("step_start_time", time.time()),
                result_summary=f"Step failed with error: {str(e)}",
            )
            return {
                "step_results": state.get("step_results", []) + [step_result.model_dump()],
                "current_step_index": step_index + 1,
                "step_start_time": time.time(),
            }
        else:
            executor.shutdown(wait=False)

        # Extract results
        msgs = result.get("messages", [])
        result_summary = _extract_ai_content(msgs)
        tool_calls_count = _count_tool_messages(msgs)
        elapsed = time.time() - state.get("step_start_time", time.time())

        step_result = StepResult(
            step_index=step_index,
            status="completed",
            tool_calls_count=tool_calls_count,
            elapsed_s=elapsed,
            result_summary=result_summary[:2000],
        )

        logger.info(
            f"[GRAPH:EXECUTE] Step {step_index} done in {elapsed:.1f}s "
            f"({tool_calls_count} tool calls)"
        )

        return {
            "step_results": state.get("step_results", []) + [step_result.model_dump()],
            "current_step_index": step_index + 1,
            "step_start_time": time.time(),
        }

    # ------------------------------------------------------------------
    # Conditional edge: MONITOR
    # ------------------------------------------------------------------
    def monitor_check(self, state: ResearchGraphState) -> str:
        """Pure Python routing — decides next node after each step execution.

        Routes to:
        - "execute"  → more steps to run, last step was fine
        - "replan"   → last step looks unsuccessful and we haven't replanned too much
        - "validate" → all steps done (or nothing left to try)
        """
        step_index = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        replan_count = state.get("replan_count", 0)
        step_results = state.get("step_results", [])

        # All steps completed?
        if step_index >= len(plan):
            # Check if the last step was unsuccessful and we can still replan
            if (
                step_results
                and _step_looks_unsuccessful(step_results[-1])
                and replan_count < self.max_replans
            ):
                logger.info(
                    f"[GRAPH:MONITOR] All steps done but last step unsuccessful "
                    f"→ replan (attempt {replan_count + 1}/{self.max_replans})"
                )
                return "replan"

            logger.info("[GRAPH:MONITOR] All steps completed → validate")
            return "validate"

        # Check if the step that just ran was unsuccessful
        if step_results and _step_looks_unsuccessful(step_results[-1]):
            if replan_count < self.max_replans:
                logger.info(
                    f"[GRAPH:MONITOR] Step {step_index - 1} unsuccessful "
                    f"→ replan (attempt {replan_count + 1}/{self.max_replans})"
                )
                return "replan"
            else:
                logger.info(
                    f"[GRAPH:MONITOR] Step {step_index - 1} unsuccessful "
                    f"but max replans ({self.max_replans}) reached → continue"
                )

        logger.info(f"[GRAPH:MONITOR] Step {step_index}/{len(plan)} → execute next")
        return "execute"

    # ------------------------------------------------------------------
    # Node: VALIDATE
    # ------------------------------------------------------------------
    def validate_node(self, state: ResearchGraphState) -> dict:
        """Synthesize final answer from all step results."""
        logger.info("[GRAPH:VALIDATE] Synthesizing final answer...")

        step_summaries = ""
        for sr in state.get("step_results", []):
            step_summaries += (
                f"\n### Step {sr['step_index']} (status: {sr['status']}, "
                f"tool_calls: {sr.get('tool_calls_count', 0)}, "
                f"elapsed: {sr.get('elapsed_s', 0):.1f}s)\n"
                f"{sr.get('result_summary', '(no summary)')}\n"
            )

        if not step_summaries.strip():
            step_summaries = "(No step results available)"

        prompt = VALIDATE_PROMPT.format(
            query=state["query"],
            step_summaries=step_summaries,
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Please synthesize the final answer."),
        ])

        answer = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"[GRAPH:VALIDATE] Final answer: {len(answer)} chars")

        return {"final_answer": answer}

    # ------------------------------------------------------------------
    # Build & compile
    # ------------------------------------------------------------------
    def build(self) -> StateGraph:
        """Construct and return the compiled graph."""
        graph = StateGraph(ResearchGraphState)

        # Add nodes
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("plan", self.plan_node)
        graph.add_node("execute", self.execute_node)
        graph.add_node("replan", self.replan_node)
        graph.add_node("validate", self.validate_node)

        # Edges
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("plan", "execute")

        # After execute, monitor decides: next step / replan / validate
        graph.add_conditional_edges(
            "execute",
            self.monitor_check,
            {
                "execute": "execute",
                "replan": "replan",
                "validate": "validate",
            },
        )

        # After replan, go back to executing the (revised) next step
        graph.add_edge("replan", "execute")

        graph.add_edge("validate", END)

        return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_research_graph(
    llm,
    tools: list,
    retrieve_tool_names: Optional[List[str]] = None,
    execute_tool_names: Optional[List[str]] = None,
    step_timeout_s: int = 45,
    max_step_tool_calls: int = 10,
    max_replans: int = 2,
):
    """Create a compiled research graph ready for invocation.

    Returns a compiled LangGraph that accepts:
        {"query": "...", "messages": [...]}
    and returns state with "final_answer", "plan", "step_results" etc.
    """
    builder = ResearchGraphBuilder(
        llm=llm,
        tools=tools,
        retrieve_tool_names=retrieve_tool_names,
        execute_tool_names=execute_tool_names,
        step_timeout_s=step_timeout_s,
        max_step_tool_calls=max_step_tool_calls,
        max_replans=max_replans,
    )
    return builder.build()


def get_graph_mermaid(
    llm,
    tools: list,
    **kwargs,
) -> str:
    """Return the Mermaid diagram string for the research graph.

    Useful for logging or rendering in Streamlit / notebooks.
    """
    graph = create_research_graph(llm, tools, **kwargs)
    return graph.get_graph().draw_mermaid()
