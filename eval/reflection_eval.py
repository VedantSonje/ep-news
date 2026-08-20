"""
Reflection Eval — compares retrieve() results with and without reflection.

Usage:
    python eval/reflection_eval.py
    python eval/reflection_eval.py --queries "defence orders last month" "pharma results"

Output: a side-by-side comparison table showing how reflection changes result counts,
which params were corrected, and the reason Ollama provided.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import AppConfig
from api.chat_handler import ChatHandler

# ── Test query suite ──────────────────────────────────────────────────────────
# (query, note) — note describes what reflection should fix
DEFAULT_QUERIES: list[tuple[str, str]] = [
    # ── Genuinely empty — reflection should fire and fix ──────────────────────
    # Impossibly high threshold — reflection should lower/remove it
    ("orders above 50000 Cr this month",         "threshold too high (50000 Cr)"),
    ("defence orders above 20000 Cr last month", "very high threshold + sector"),
    # Wrong/non-canonical sector name — reflection should fix spelling
    ("militery sector orders this month",        "typo: militery → aerospace & defence"),
    ("pharmceutical results this month",         "typo: pharmceutical → healthcare"),
    ("infra orders above 15000 Cr last week",    "high threshold + narrow date"),
    # Very specific past date with no data
    ("results on 2026-01-01",                    "holiday, no data — widen date"),
    ("orders on 2026-01-26",                     "Republic Day holiday"),
    # Narrow date window on obscure sector
    ("cement sector orders last 2 days",         "narrow date + less-active sector"),
    ("textile orders last 3 days",               "narrow date + infrequent sector"),
    # ── Control queries — should work fine both ways ──────────────────────────
    ("pharma results this month",                "control: broad + active sector"),
    ("order wins this month",                    "control: always has data"),
    ("INFY results last year",                   "control: company-specific"),
]


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _sym_list(results: list[dict], n: int = 5) -> str:
    syms = [r.get("symbol", "?") for r in results[:n]]
    tail = f" +{len(results)-n} more" if len(results) > n else ""
    return ", ".join(syms) + tail


def _reflection_info(results: list[dict]) -> str:
    for r in results:
        if r.get("_reflected"):
            return r.get("_reflection_reason", "reflected")
    return "—"


def _bar(n: int, width: int = 20) -> str:
    filled = min(n, width)
    return "█" * filled + "░" * (width - filled)


# ── Main eval ─────────────────────────────────────────────────────────────────

def run_eval(queries: list[tuple[str, str]], handler: ChatHandler) -> None:
    COL_Q   = 45
    COL_N   = 8
    COL_BAR = 22
    COL_SYM = 32
    COL_REF = 40

    header = (
        f"{'Query':<{COL_Q}} {'No Reflect':>{COL_N}} {'Reflect':>{COL_N}} "
        f"{'Delta':>6}  {'Reflection reason':<{COL_REF}}"
    )
    sep = "─" * len(header)
    print("\n" + "═" * len(header))
    print("  REFLECTION EVAL — before vs after")
    print("═" * len(header))
    print(header)
    print(sep)

    wins = 0
    total_with_reflection = 0
    total_without_reflection = 0

    for query, note in queries:
        # Without reflection
        t0 = time.perf_counter()
        results_no, intent_no, min_cr_no = handler.retrieve(query, reflect=False)
        t_no = time.perf_counter() - t0

        # With reflection
        t1 = time.perf_counter()
        results_rf, intent_rf, min_cr_rf = handler.retrieve(query, reflect=True)
        t_rf = time.perf_counter() - t1

        n_no = len(results_no)
        n_rf = len(results_rf)
        delta = n_rf - n_no
        total_without_reflection += n_no
        total_with_reflection    += n_rf

        ref_reason = _reflection_info(results_rf)
        reflected  = any(r.get("_reflected") for r in results_rf)

        delta_str = (
            f"\033[92m+{delta}\033[0m" if delta > 0
            else (f"\033[91m{delta}\033[0m" if delta < 0 else "  0")
        )
        reflect_marker = " ✓" if reflected else "  "

        q_display = query if len(query) <= COL_Q else query[:COL_Q-1] + "…"
        print(
            f"{q_display:<{COL_Q}} {n_no:>{COL_N}} {n_rf:>{COL_N}} "
            f"{delta_str:>6}{reflect_marker}  {ref_reason:<{COL_REF}}"
        )
        # Detail row: intent + sample symbols
        intent_str = f"intent={intent_no}" + (f"→{intent_rf}" if intent_rf != intent_no else "")
        syms_no = _sym_list(results_no) if results_no else "—"
        syms_rf = _sym_list(results_rf) if results_rf else "—"
        print(f"  [{note}]")
        print(f"  {intent_str}  | no-reflect: {syms_no}")
        if reflected:
            print(f"  reflect symbols : {_sym_list(results_rf)}")
        print()

        if delta > 0:
            wins += 1

    print(sep)
    print(f"  Queries where reflection helped : {wins} / {len(queries)}")
    print(f"  Total results without reflection : {total_without_reflection}")
    print(f"  Total results with reflection    : {total_with_reflection}")
    print(f"  Net gain from reflection         : +{total_with_reflection - total_without_reflection}")
    print("═" * len(header) + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Reflection eval")
    parser.add_argument(
        "--queries", nargs="+",
        help="Custom queries to test (overrides default suite)",
    )
    args = parser.parse_args()

    cfg     = AppConfig.from_env()
    handler = ChatHandler(chroma_path=cfg.chroma_path, db_path=cfg.db_path)

    if args.queries:
        queries = [(q, "custom") for q in args.queries]
    else:
        queries = DEFAULT_QUERIES

    print(f"Running reflection eval on {len(queries)} queries…")
    print("(Reflection round-trips to Ollama — expect a few seconds per empty result)\n")
    run_eval(queries, handler)


if __name__ == "__main__":
    main()
