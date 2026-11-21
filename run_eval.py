"""
Runs the full agent against eval/benchmark.json and produces real, reportable
numbers: accuracy, latency, tool-call counts, and a failure-mode breakdown.

This is the piece that turns "I built an agent" into "I measured an agent" —
run this after setup and the printed summary + eval/report.json are what you
quote in your CV / show in an interview.

Usage:
    python -m eval.run_eval --model qwen3:8b
    python -m eval.run_eval --model qwen3:8b --limit 3   # quick smoke test
"""

from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path

from server import data as market_data
from server import analytics
from client.ollama_bridge import run_agent_sync, DEFAULT_MODEL, AgentTimeoutError

BENCH_PATH = Path(__file__).parent / "benchmark.json"
REPORT_PATH = Path(__file__).parent / "report.json"

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _get_nested(d: dict, dotted_path: str):
    cur = d
    for key in dotted_path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur


def _compute_oracle(question: dict, period_default: str) -> tuple:
    """Compute ground truth by calling the same analytics functions the agent's tools use."""
    tickers = question["oracle_tickers"]
    period = question.get("period", period_default)
    fn = question["oracle_fn"]

    if fn == "company_sector":
        info = market_data.get_company_info(tickers[0])
        oracle_dict = info
    elif fn in ("compare_stocks", "correlation_matrix"):
        dfs = {t: market_data.get_price_history(t, period=period) for t in tickers}
        oracle_dict = getattr(analytics, fn)(dfs)
    else:
        df = market_data.get_price_history(tickers[0], period=period)
        oracle_dict = getattr(analytics, fn)(df)

    value = _get_nested(oracle_dict, question["oracle_field"])
    return value, oracle_dict


def _extract_number_near(text: str) -> list[float]:
    matches = NUMBER_RE.findall(text)
    out = []
    for m in matches:
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _score_question(question: dict, oracle_value, agent_answer: str, default_tol: float) -> dict:
    match_type = question.get("match_type", "numeric")
    tol = question.get("tolerance_pct", default_tol)

    if match_type == "string_contains":
        correct = str(oracle_value).lower() in agent_answer.lower()
        return {"correct": correct, "oracle": oracle_value, "matched_value": None, "match_type": match_type}

    if oracle_value is None:
        return {"correct": False, "oracle": None, "matched_value": None, "match_type": match_type,
                "note": "oracle value unavailable (insufficient history for this window?)"}

    candidates = _extract_number_near(agent_answer)
    if not candidates:
        return {"correct": False, "oracle": oracle_value, "matched_value": None, "match_type": match_type,
                "note": "agent gave no extractable number"}

    if tol == 0:
        best = min(candidates, key=lambda c: abs(c - oracle_value))
        correct = best == oracle_value
    else:
        denom = max(abs(oracle_value), 1e-9)
        best = min(candidates, key=lambda c: abs(c - oracle_value) / denom * 100)
        correct = (abs(best - oracle_value) / denom * 100) <= tol

    return {"correct": correct, "oracle": oracle_value, "matched_value": best,
            "tolerance_pct": tol, "match_type": match_type}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--think", action="store_true",
                         help="Enable Qwen3's thinking mode (much slower, esp. on CPU; off by default here for speed)")
    parser.add_argument("--timeout", type=float, default=240.0,
                         help="Max seconds to wait per question before giving up and moving on (default: 240)")
    parser.add_argument("--verbose", action="store_true",
                         help="Print live progress (subprocess start, tool discovery, heartbeat every 10s) — use this to diagnose a slow/stuck run")
    parser.add_argument("--output", default=None,
                         help="Output report filename, saved inside eval/ (default: report.json). "
                              "Use a distinct name per run (e.g. report_think_on.json) when comparing "
                              "configurations, or each run silently overwrites the last one's results.")
    args = parser.parse_args()

    report_path = (Path(__file__).parent / args.output) if args.output else REPORT_PATH

    bench = json.loads(BENCH_PATH.read_text())
    questions = bench["questions"][: args.limit] if args.limit else bench["questions"]
    default_tol = bench.get("tolerance_pct_default", 3.0)
    period_default = bench.get("period_default", "6mo")

    results = []
    print(f"Running {len(questions)} benchmark questions against model='{args.model}'...\n")

    for q in questions:
        print(f"[{q['id']}] {q['query']}", flush=True)
        try:
            oracle_value, oracle_full = _compute_oracle(q, period_default)
        except Exception as e:
            print(f"  !! could not compute oracle (data fetch failed): {e}", flush=True)
            results.append({"id": q["id"], "query": q["query"], "correct": False,
                             "error": f"oracle_fetch_failed: {e}"})
            continue

        print(f"  -> calling agent (timeout {args.timeout}s)...", flush=True)
        t0 = time.monotonic()
        try:
            agent_result = run_agent_sync(
                q["query"], model=args.model, ollama_host=args.ollama_host,
                think=args.think, timeout_s=args.timeout, verbose=args.verbose,
            )
        except AgentTimeoutError as e:
            elapsed = round(time.monotonic() - t0, 2)
            print(f"  -> TIMEOUT after {elapsed}s — {e}\n", flush=True)
            partial = e.partial_result
            results.append({
                "id": q["id"], "query": q["query"], "correct": False,
                "error": str(e), "elapsed_s": elapsed, "timed_out": True,
                "iterations_reached": getattr(partial, "iterations", 0),
                "tool_calls_before_timeout": [tc.name for tc in getattr(partial, "tool_calls", [])],
            })
            continue
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 2)
            print(f"  -> ERROR after {elapsed}s: {e}\n", flush=True)
            results.append({
                "id": q["id"], "query": q["query"], "correct": False,
                "error": str(e), "elapsed_s": elapsed,
            })
            continue
        elapsed = round(time.monotonic() - t0, 2)

        score = _score_question(q, oracle_value, agent_result.final_answer, default_tol)
        results.append({
            "id": q["id"],
            "query": q["query"],
            **score,
            "agent_answer": agent_result.final_answer,
            "num_tool_calls": len(agent_result.tool_calls),
            "tool_calls": [tc.name for tc in agent_result.tool_calls],
            "tool_errors": [tc.error for tc in agent_result.tool_calls if tc.error],
            "iterations": agent_result.iterations,
            "hit_iteration_cap": agent_result.hit_iteration_cap,
            "elapsed_s": elapsed,
        })
        mark = "PASS" if score["correct"] else "FAIL"
        print(f"  -> {mark}  (oracle={score.get('oracle')}, agent said~{score.get('matched_value')}, {elapsed}s, {len(agent_result.tool_calls)} tool calls)\n")

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    avg_latency = round(sum(r.get("elapsed_s", 0) for r in results) / n, 2) if n else 0
    avg_tool_calls = round(sum(r.get("num_tool_calls", 0) for r in results) / n, 2) if n else 0
    cap_hits = sum(1 for r in results if r.get("hit_iteration_cap"))
    tool_error_qs = sum(1 for r in results if r.get("tool_errors"))

    summary = {
        "model": args.model,
        "thinking_mode": args.think,
        "total_questions": n,
        "correct": n_correct,
        "accuracy_pct": round(100 * n_correct / n, 1) if n else 0,
        "avg_latency_s": avg_latency,
        "avg_tool_calls_per_query": avg_tool_calls,
        "iteration_cap_hits": cap_hits,
        "questions_with_tool_errors": tool_error_qs,
    }

    report = {"summary": summary, "results": results}
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
