"""
The agent itself: bridges a local Ollama model to the FinMCP tool server.

Ollama's Python client does not speak MCP natively (there's an open feature
request for it), so this module does the translation by hand:

  1. Launch the MCP server as a subprocess, connect over stdio.
  2. List its tools, convert each MCP tool schema -> Ollama's OpenAI-style
     function-calling schema.
  3. Run the standard tool-calling loop: send the conversation + tool list to
     Ollama, if it returns tool_calls execute them against the live MCP
     session, feed the results back, repeat until it returns a final answer
     or we hit the iteration cap.

Every step is logged into an AgentResult so the eval harness (eval/run_eval.py)
can inspect exactly what tools were called, with what arguments, and how long
each turn took — not just the final text.
"""

from __future__ import annotations
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM_PROMPT = """You are a finance data analyst agent. You have tools that fetch real \
market data and compute real statistics — you do not have this data memorized and must \
never guess or estimate a number. For any question involving a price, return, moving \
average, volatility figure, or comparison, call the appropriate tool and base your answer \
only on what it returns. Cite the actual numbers from the tool result in your final answer. \
If a tool errors (e.g. invalid ticker), say so plainly rather than making up a substitute \
answer. Keep answers concise and quantitative."""

MAX_ITERATIONS = 6
DEFAULT_MODEL = "qwen3:8b"


@dataclass
class ToolCallLog:
    name: str
    arguments: dict
    result_preview: str
    elapsed_s: float
    error: str | None = None


@dataclass
class AgentResult:
    query: str
    final_answer: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    iterations: int = 0
    total_elapsed_s: float = 0.0
    hit_iteration_cap: bool = False


def _mcp_tool_to_ollama(tool) -> dict:
    """Convert an MCP Tool object into Ollama/OpenAI-style function schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def _extract_text(call_tool_result) -> str:
    """Pull plain text out of an MCP CallToolResult (list of content blocks)."""
    parts = []
    for block in call_tool_result.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts) if parts else json.dumps(call_tool_result.structured_content)


async def _chat_with_heartbeat(client, model, messages, tools, think, verbose, heartbeat_s=10.0):
    """
    Await client.chat() but print a heartbeat line every heartbeat_s seconds
    while waiting, so it's possible to see live whether Ollama is just slow
    or has actually stopped responding — instead of staring at a silent
    terminal for 4 minutes with no idea which one it is.
    """
    task = asyncio.ensure_future(client.chat(model=model, messages=messages, tools=tools, think=think))
    waited = 0.0
    while not task.done():
        await asyncio.wait([task], timeout=heartbeat_s)
        waited += heartbeat_s
        if not task.done() and verbose:
            print(f"      ... still waiting on Ollama ({waited:.0f}s so far)", flush=True)
    return await task


async def run_agent(
    query: str,
    model: str = DEFAULT_MODEL,
    server_command: str | None = None,
    server_args: list[str] | None = None,
    ollama_host: str | None = None,
    think: bool = False,
    verbose: bool = False,
    _result_holder: dict | None = None,
) -> AgentResult:
    """
    Run one query through the full agent loop: MCP server subprocess + local
    Ollama model doing multi-step tool calling.

    think: whether to enable the model's extended reasoning ("thinking mode").
    Qwen3 ships with this ON by default in Ollama, which can add hundreds to
    thousands of reasoning tokens to EVERY turn (tool-selection turns too, not
    just the final answer) — on CPU-only hardware this is the difference
    between a few seconds and several minutes per question. Defaults to False
    here since this is a well-specified tool-calling task where the speed/
    quality tradeoff strongly favors speed; pass think=True if you want to
    measure whether it actually improves accuracy on your benchmark (a good,
    genuinely interesting thing to report either way).

    verbose: print live progress (subprocess start, tool discovery, and a
    heartbeat every 10s while waiting on each Ollama call) so a slow run is
    visibly "working" instead of silent.
    """
    # Default to sys.executable, NOT a hardcoded "python3" — on Windows there
    # is usually no "python3" on PATH (only "python"), and even where there
    # is, it may resolve to a different interpreter than the one running this
    # process (and therefore not have the venv's packages installed). Using
    # sys.executable guarantees the subprocess is launched with the exact
    # same interpreter/environment as the parent.
    server_command = server_command or sys.executable
    server_args = server_args or ["-m", "server.finance_server"]
    client = ollama.AsyncClient(host=ollama_host) if ollama_host else ollama.AsyncClient()

    result = AgentResult(query=query, final_answer="")
    if _result_holder is not None:
        # Expose this exact (mutable) object to the caller immediately, so that
        # even if this coroutine is cancelled (e.g. by a timeout) partway
        # through, whatever progress was made — iterations reached, tool
        # calls completed so far — is still visible to whoever's holding
        # this dict, instead of being silently lost with the cancelled task.
        _result_holder["result"] = result
    t_start = time.monotonic()
    if verbose:
        print(f"    [t=0.0s] launching MCP server subprocess...", flush=True)

    params = StdioServerParameters(command=server_command, args=server_args, cwd=".")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if verbose:
                print(f"    [t={time.monotonic()-t_start:.1f}s] MCP session initialized, listing tools...", flush=True)
            mcp_tools = (await session.list_tools()).tools
            ollama_tools = [_mcp_tool_to_ollama(t) for t in mcp_tools]
            if verbose:
                print(f"    [t={time.monotonic()-t_start:.1f}s] got {len(mcp_tools)} tools, calling Ollama ({model}, think={think})...", flush=True)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]

            for iteration in range(1, MAX_ITERATIONS + 1):
                result.iterations = iteration
                response = await _chat_with_heartbeat(client, model, messages, ollama_tools, think, verbose)
                if verbose:
                    print(f"    [t={time.monotonic()-t_start:.1f}s] Ollama responded (iteration {iteration})", flush=True)
                msg = response.message

                if not msg.tool_calls:
                    result.final_answer = msg.content or "(model returned no content)"
                    break

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })

                for call in msg.tool_calls:
                    t0 = time.monotonic()
                    name = call.function.name
                    args = call.function.arguments or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    error = None
                    try:
                        tool_result = await session.call_tool(name, args)
                        text = _extract_text(tool_result)
                        if getattr(tool_result, "is_error", False):
                            error = text  # tool ran but reported failure (e.g. bad ticker)
                    except Exception as e:
                        text = f"ERROR calling {name}: {e}"
                        error = str(e)

                    elapsed = time.monotonic() - t0
                    result.tool_calls.append(ToolCallLog(
                        name=name, arguments=args,
                        result_preview=text[:300],
                        elapsed_s=round(elapsed, 3),
                        error=error,
                    ))
                    messages.append({"role": "tool", "content": text})
            else:
                result.hit_iteration_cap = True
                result.final_answer = (
                    f"(stopped after {MAX_ITERATIONS} tool-call rounds without a final answer — "
                    f"likely a tool-selection loop; see tool_calls log)"
                )

    result.total_elapsed_s = round(time.monotonic() - t_start, 2)
    return result


def _describe_exception(exc: BaseException) -> str:
    """
    Unwrap Python's ExceptionGroup/TaskGroup wrapping to find the actual root
    cause. asyncio TaskGroups (used internally by the MCP stdio transport)
    wrap any sub-task failure in an ExceptionGroup, whose default str() is
    the unhelpful "unhandled errors in a TaskGroup (N sub-exceptions)" — this
    walks down to the real exception(s) so errors are actually diagnosable.
    """
    excs = getattr(exc, "exceptions", None)  # ExceptionGroup / BaseExceptionGroup
    if not excs:
        return f"{type(exc).__name__}: {exc}"
    return " | ".join(_describe_exception(e) for e in excs)


class AgentTimeoutError(TimeoutError):
    """Raised when a run exceeds its timeout. Carries whatever partial progress
    was made (iterations reached, tool calls completed) via .partial_result,
    so callers can report *where* it got stuck instead of just *that* it did."""
    def __init__(self, message: str, partial_result: AgentResult | None = None):
        super().__init__(message)
        self.partial_result = partial_result


def run_agent_sync(query: str, timeout_s: float | None = None, **kwargs) -> AgentResult:
    """
    Convenience sync wrapper for non-async callers (e.g. Streamlit, the eval harness).
    Accepts the same kwargs as run_agent (model, think, verbose, ollama_host, etc.),
    plus an optional wall-clock timeout_s so a single stuck/looping question can't
    hang forever.
    """
    holder: dict = {}

    async def _run():
        coro = run_agent(query, _result_holder=holder, **kwargs)
        if timeout_s:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return await coro

    try:
        return asyncio.run(_run())
    except asyncio.TimeoutError as e:
        partial = holder.get("result")
        if partial is not None and (partial.iterations or partial.tool_calls):
            detail = (
                f" Reached iteration {partial.iterations}/{MAX_ITERATIONS} with "
                f"{len(partial.tool_calls)} tool call(s) completed before the timeout."
            )
        else:
            detail = (
                " Never completed even the first model response — the very first "
                "call to Ollama didn't return anything within the timeout."
            )
        raise AgentTimeoutError(
            f"Agent run exceeded {timeout_s}s without finishing.{detail}",
            partial_result=partial,
        ) from e
    except BaseException as e:
        # Re-raise with the real cause surfaced instead of a generic
        # "unhandled errors in a TaskGroup" message.
        raise RuntimeError(f"Agent run failed: {_describe_exception(e)}") from e


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    q = " ".join(args) or "What's AAPL's 20-day moving average and how volatile has it been?"
    res = run_agent_sync(q, verbose=verbose)
    print(f"\nQuery: {res.query}")
    print(f"Iterations: {res.iterations}  |  Total time: {res.total_elapsed_s}s")
    print(f"\nTool calls made ({len(res.tool_calls)}):")
    for tc in res.tool_calls:
        status = f"ERROR: {tc.error}" if tc.error else "ok"
        print(f"  - {tc.name}({tc.arguments}) [{tc.elapsed_s}s, {status}]")
    print(f"\nFinal answer:\n{res.final_answer}")
