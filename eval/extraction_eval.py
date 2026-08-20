"""
Extraction Eval — scores financial PDF extraction accuracy against fixture ground truth.

Four scored dimensions:
  1. numeric_accuracy   — revenue_cr, pat_cr, eps, ebitda_cr within 2% of expected
  2. unit_conversion    — detects Lakhs/Crores confusion (off by ~100x)
  3. period_match       — extracted period string matches expected
  4. confidence_calib   — confidence field matches actual field-fill quality

Usage:
    # Debug a single bad PDF — no fixture needed:
    python eval/extraction_eval.py --pdf URL --symbol TCS --company "Tata Consultancy"
    python eval/extraction_eval.py --pdf /path/to/file.pdf --symbol TCS --company "TCS"

    # With known expected values (scores the result):
    python eval/extraction_eval.py --pdf URL --symbol TCS --company "TCS" ^
        --expected-revenue 63437 --expected-pat 12760 --expected-eps 34.69 ^
        --expected-period "Q1 FY2027"

    # Save a debugged PDF as a new fixture for future regression runs:
    python eval/extraction_eval.py --pdf URL --symbol TCS --company "TCS" --save-fixture

    # Run fixture suite:
    python eval/extraction_eval.py                       # run all fixtures
    python eval/extraction_eval.py --verified-only       # skip unverified fixtures
    python eval/extraction_eval.py --seed-from-db 10     # seed JSONL from DB records
    python eval/extraction_eval.py --no-langsmith        # skip LangSmith logging
    python eval/extraction_eval.py --fixture path.jsonl  # custom fixture file
    python eval/extraction_eval.py --tol 0.05            # numeric tolerance (default 2%)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from config import AppConfig

DEFAULT_FIXTURES = ROOT / "eval" / "fixtures" / "financial_fixtures.jsonl"
_NUMERIC_KEYS = ("revenue_cr", "pat_cr", "eps", "ebitda_cr", "ebitda_margin_pct")

# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class NumericScore:
    key:       str
    expected:  float | None
    extracted: float | None
    passed:    bool
    pct_err:   float | None   # None if expected/extracted is None
    note:      str = ""

@dataclass
class FixtureScore:
    fixture_id:      str
    symbol:          str
    period:          str
    numeric:         list[NumericScore] = field(default_factory=list)
    period_match:    bool | None = None
    unit_ok:         bool | None = None
    confidence_ok:   bool | None = None
    confidence_got:  str | None = None
    confidence_exp:  str | None = None
    error:           str | None = None    # download/extraction failure

    @property
    def numeric_pass_rate(self) -> float:
        tested = [s for s in self.numeric if s.expected is not None]
        if not tested:
            return 1.0
        return sum(1 for s in tested if s.passed) / len(tested)

    @property
    def overall_pass(self) -> bool:
        if self.error:
            return False
        checks = [s.passed for s in self.numeric if s.expected is not None]
        if self.period_match is not None:
            checks.append(self.period_match)
        if self.confidence_ok is not None:
            checks.append(self.confidence_ok)
        return bool(checks) and all(checks)


def _pct_err(extracted: float, expected: float) -> float:
    if expected == 0:
        return 0.0 if extracted == 0 else float("inf")
    return abs(extracted - expected) / abs(expected)


def score_numerics(result: dict, expected: dict, tol: float) -> list[NumericScore]:
    scores = []
    for key in _NUMERIC_KEYS:
        exp_val = expected.get(key)
        got_val = result.get(key)

        if exp_val is None:
            # No ground truth — record as informational
            scores.append(NumericScore(key=key, expected=None, extracted=got_val,
                                       passed=True, pct_err=None, note="no ground truth"))
            continue

        if got_val is None:
            scores.append(NumericScore(key=key, expected=exp_val, extracted=None,
                                       passed=False, pct_err=None, note="field missing"))
            continue

        pct = _pct_err(float(got_val), float(exp_val))
        passed = pct <= tol
        scores.append(NumericScore(
            key=key, expected=exp_val, extracted=float(got_val),
            passed=passed, pct_err=pct,
            note=f"{pct*100:.1f}% error" if not passed else "",
        ))
    return scores


def score_unit_conversion(result: dict, fixture: dict) -> tuple[bool, str]:
    """
    Detect Lakhs/Crores confusion: if extracted value is ~100x off from expected,
    the model failed to divide by 100.
    """
    declared = fixture.get("declared_unit", "").lower()
    expected = fixture.get("expected", {})

    for key in ("revenue_cr", "pat_cr"):
        exp = expected.get(key)
        got = result.get(key)
        if exp is None or got is None or exp == 0:
            continue
        ratio = abs(float(got)) / abs(float(exp))
        # ~100x off → Lakhs not converted
        if 80 < ratio < 120:
            return False, f"{key}: got ~{got:.1f} but expected ~{exp:.1f} (100x off — Lakhs not converted)"
        # ~10x off → Millions not converted
        if 8 < ratio < 12:
            return False, f"{key}: got ~{got:.1f} but expected ~{exp:.1f} (10x off — Millions not converted)"
    return True, "ok"


def score_period(result: dict, expected: dict) -> tuple[bool, str]:
    exp_period = expected.get("period", "")
    got_period = str(result.get("period") or "")
    if not exp_period:
        return True, "no expected period"
    match = got_period.strip() == exp_period.strip()
    return match, f"got '{got_period}' expected '{exp_period}'"


def score_confidence(result: dict) -> tuple[bool, str, str]:
    """
    Check if the confidence field matches actual fill quality.
    Returns (passed, expected_conf, got_conf).
    """
    filled = sum(1 for k in ("revenue_cr", "pat_cr", "ebitda_cr", "eps")
                 if result.get(k) is not None)
    if filled >= 3:
        expected_conf = "high"
    elif filled >= 1:
        expected_conf = "medium"
    else:
        expected_conf = "low"

    got_conf = str(result.get("confidence") or "")
    passed = got_conf == expected_conf
    return passed, expected_conf, got_conf


# ── Extraction ─────────────────────────────────────────────────────────────────

def _fetch_pdf(pdf_url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (compatible; EP-News-Eval/1.0)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _result_to_dict(result_obj) -> dict:
    return {
        "period":            result_obj.period,
        "revenue_cr":        result_obj.revenue_cr,
        "pat_cr":            result_obj.pat_cr,
        "ebitda_cr":         result_obj.ebitda_cr,
        "ebitda_margin_pct": result_obj.ebitda_margin_pct,
        "eps":               result_obj.eps,
        "confidence":        result_obj.confidence,
        "raw_summary":       result_obj.raw_summary,
        "key_highlights":    result_obj.key_highlights,
    }


def _extract_fixture(fixture: dict, force_vision: bool = False) -> dict | None:
    """Download PDF and run through LocalExtractor (or vision-only). Returns raw result dict or None."""
    pdf_url      = fixture["pdf_url"]
    symbol       = fixture["symbol"]
    company      = fixture["company"]
    broadcast_dt = fixture.get("broadcast_dt", "2026-01-01") + " 00:00"
    period_type  = fixture.get("period_type", "quarterly")
    etype        = period_type if period_type in ("quarterly", "annual") else "financial_results"

    try:
        pdf_bytes = _fetch_pdf(pdf_url)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    if force_vision:
        from financial.vision_extractor import extract_with_vision
        from financial.local_extractor import LocalExtractor
        print(f"    [vision] forcing Gemini Flash path", flush=True)
        data = extract_with_vision(pdf_bytes, symbol, company, broadcast_dt, pdf_url, "financial_results")
        if data is None:
            return None
        extractor = LocalExtractor()
        result_obj = extractor._build_result(data, symbol, company, broadcast_dt, pdf_url, etype)
        return _result_to_dict(result_obj) if result_obj else None

    from financial.local_extractor import LocalExtractor
    extractor  = LocalExtractor()
    result_obj = extractor.extract(
        pdf_bytes=pdf_bytes, symbol=symbol, company=company,
        broadcast_dt=broadcast_dt, source_url=pdf_url, extraction_type=etype,
    )
    return _result_to_dict(result_obj) if result_obj else None


# ── LangSmith logging ─────────────────────────────────────────────────────────

def _langsmith_log(fixture: dict, result: dict | None, score: FixtureScore) -> None:
    try:
        from langsmith import Client
        client = Client()

        project = os.getenv("LANGSMITH_PROJECT", "ep-news-ai")
        run_id  = f"eval-{fixture['id']}-{int(time.time())}"

        inputs = {
            "fixture_id":  fixture["id"],
            "symbol":      fixture["symbol"],
            "period":      fixture.get("expected", {}).get("period", ""),
            "pdf_url":     fixture["pdf_url"],
            "declared_unit": fixture.get("declared_unit", ""),
        }
        outputs = {
            "extracted":   result or {},
            "expected":    fixture.get("expected", {}),
        }
        feedback = {
            "numeric_pass_rate":  score.numeric_pass_rate,
            "period_match":       int(score.period_match) if score.period_match is not None else None,
            "unit_ok":            int(score.unit_ok) if score.unit_ok is not None else None,
            "confidence_ok":      int(score.confidence_ok) if score.confidence_ok is not None else None,
            "overall_pass":       int(score.overall_pass),
        }

        client.create_run(
            name="financial_extraction_eval",
            run_type="chain",
            project_name=project,
            inputs=inputs,
            outputs={**outputs, "scores": feedback},
            tags=["extraction-eval", fixture["symbol"]],
        )
    except Exception as e:
        print(f"    [langsmith] logging skipped: {e}", flush=True)


# ── Seed from DB ──────────────────────────────────────────────────────────────

# Values that appear verbatim in the prompt examples — hallucination fingerprints
_HALLUCINATION_FINGERPRINTS = {1234.5, 234.5, 189.3, 106.18, 1266.42, 184.5, 2942.7}

def seed_from_db(db_path: Path, n: int, fixture_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT symbol, company, period, period_type, broadcast_dt, source_url,
               revenue_cr, pat_cr, eps, ebitda_margin_pct, ebitda_cr
        FROM financial_results
        WHERE revenue_cr IS NOT NULL AND pat_cr IS NOT NULL
          AND source_url LIKE '%.pdf'
        ORDER BY broadcast_dt DESC
        LIMIT ?
    """, (n * 3,)).fetchall()
    conn.close()

    existing_ids = set()
    if fixture_path.exists():
        for line in fixture_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    added = 0
    with fixture_path.open("a", encoding="utf-8") as f:
        for row in rows:
            symbol, company, period, ptype, bdt, url, rev, pat, eps, ebm, ebc = row
            fix_id = f"{symbol}_{period.replace(' ', '').replace('/', '')}"

            if fix_id in existing_ids:
                continue

            # Skip suspected hallucinations
            if rev in _HALLUCINATION_FINGERPRINTS or pat in _HALLUCINATION_FINGERPRINTS:
                print(f"  SKIP {fix_id} — suspected hallucination values (rev={rev}, pat={pat})")
                continue

            expected: dict[str, Any] = {"period": period}
            if rev  is not None: expected["revenue_cr"]        = round(float(rev), 2)
            if pat  is not None: expected["pat_cr"]            = round(float(pat), 2)
            if eps  is not None: expected["eps"]               = round(float(eps), 3)
            if ebm  is not None: expected["ebitda_margin_pct"] = round(float(ebm), 1)
            if ebc  is not None: expected["ebitda_cr"]         = round(float(ebc), 2)

            fixture = {
                "id":           fix_id,
                "symbol":       symbol,
                "company":      company,
                "period":       period,
                "period_type":  ptype,
                "broadcast_dt": bdt[:10],
                "pdf_url":      url,
                "declared_unit": "unknown",
                "expected":     expected,
                "verified":     False,
                "notes":        "Seeded from DB — verify against source PDF before trusting",
            }
            f.write(json.dumps(fixture) + "\n")
            existing_ids.add(fix_id)
            added += 1

            if added >= n:
                break

    print(f"Seeded {added} new fixtures → {fixture_path}")


# ── Report ─────────────────────────────────────────────────────────────────────

def _fmt_num(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:,.{decimals}f}"


def print_report(scores: list[FixtureScore]) -> None:
    W = 90
    print("\n" + "═" * W)
    print("  FINANCIAL EXTRACTION EVAL")
    print("═" * W)

    total = len(scores)
    passed = sum(1 for s in scores if s.overall_pass)
    errored = sum(1 for s in scores if s.error)

    for sc in scores:
        status = "✓" if sc.overall_pass else ("ERR" if sc.error else "✗")
        print(f"\n[{status}] {sc.fixture_id}")

        if sc.error:
            print(f"       ERROR: {sc.error}")
            continue

        # Numeric scores
        for ns in sc.numeric:
            if ns.expected is None:
                continue
            mark  = "✓" if ns.passed else "✗"
            got   = _fmt_num(ns.extracted)
            exp   = _fmt_num(ns.expected)
            pct   = f"  ({ns.pct_err*100:.1f}% err)" if ns.pct_err is not None else ""
            print(f"       {mark} {ns.key:<22} got={got:>10}  exp={exp:>10}{pct}")

        # Period
        if sc.period_match is not None:
            mark = "✓" if sc.period_match else "✗"
            print(f"       {mark} period_match")

        # Unit conversion
        if sc.unit_ok is not None:
            mark = "✓" if sc.unit_ok else "✗"
            print(f"       {mark} unit_conversion")

        # Confidence calibration
        if sc.confidence_ok is not None:
            mark = "✓" if sc.confidence_ok else "✗"
            print(f"       {mark} confidence  got='{sc.confidence_got}'  expected='{sc.confidence_exp}'")

        nr = sc.numeric_pass_rate
        bar_filled = int(nr * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"       numeric pass rate: [{bar}] {nr*100:.0f}%")

    print("\n" + "─" * W)
    print(f"  Results: {passed}/{total} fixtures fully passed  |  {errored} errored")
    if total:
        avg_nr = sum(s.numeric_pass_rate for s in scores) / total
        print(f"  Avg numeric pass rate: {avg_nr*100:.1f}%")
    print("═" * W + "\n")


# ── Single-PDF debug mode ─────────────────────────────────────────────────────

def run_single_pdf(
    pdf_source: str,   # URL or local file path
    symbol: str,
    company: str,
    broadcast_dt: str = "",
    expected: dict | None = None,
    tol: float = 0.02,
    save_fixture: bool = False,
    fixture_path: Path = DEFAULT_FIXTURES,
    force_vision: bool = False,
) -> None:
    """
    Extract a single PDF and print a detailed debug report.
    Optionally scores against expected values and saves as a fixture.
    """
    from financial.local_extractor import LocalExtractor

    broadcast_dt = broadcast_dt or (__import__("datetime").date.today().isoformat() + " 00:00")
    if len(broadcast_dt) == 10:
        broadcast_dt += " 00:00"

    W = 80
    print("\n" + "═" * W)
    print(f"  PDF EXTRACTION DEBUG — {symbol}  ({company})")
    print(f"  Source : {pdf_source}")
    print("═" * W)

    # ── Fetch PDF bytes ───────────────────────────────────────────────────────
    if pdf_source.startswith("http://") or pdf_source.startswith("https://"):
        import urllib.request
        print("\nDownloading PDF…", flush=True)
        try:
            req = urllib.request.Request(pdf_source, headers={
                "User-Agent": "Mozilla/5.0 (compatible; EP-News-Eval/1.0)"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                pdf_bytes = resp.read()
            print(f"  {len(pdf_bytes)//1024} KB downloaded", flush=True)
        except Exception as e:
            print(f"  ERROR downloading PDF: {e}")
            return
    else:
        p = Path(pdf_source)
        if not p.exists():
            print(f"  ERROR: file not found: {pdf_source}")
            return
        pdf_bytes = p.read_bytes()
        print(f"  {len(pdf_bytes)//1024} KB loaded from disk", flush=True)

    # ── Extract ───────────────────────────────────────────────────────────────
    src_url = pdf_source if pdf_source.startswith("http") else f"file://{pdf_source}"
    pipeline_label = "Gemini Flash (vision)" if force_vision else "full pipeline"
    print(f"\nRunning {pipeline_label}…", flush=True)
    t0 = time.perf_counter()

    if force_vision:
        from financial.vision_extractor import extract_with_vision
        extractor = LocalExtractor()
        data = extract_with_vision(pdf_bytes, symbol, company, broadcast_dt, src_url, "financial_results")
        result_obj = extractor._build_result(data, symbol, company, broadcast_dt, src_url, "financial_results") if data else None
    else:
        extractor  = LocalExtractor()
        result_obj = extractor.extract(
            pdf_bytes=pdf_bytes, symbol=symbol, company=company,
            broadcast_dt=broadcast_dt, source_url=src_url, extraction_type="financial_results",
        )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s", flush=True)

    if result_obj is None:
        print("\n  RESULT: extractor returned None — likely a scanned PDF or Gemini unavailable")
        return

    result = {
        "period":            result_obj.period,
        "revenue_cr":        result_obj.revenue_cr,
        "pat_cr":            result_obj.pat_cr,
        "ebitda_cr":         result_obj.ebitda_cr,
        "ebitda_margin_pct": result_obj.ebitda_margin_pct,
        "eps":               result_obj.eps,
        "confidence":        result_obj.confidence,
        "raw_summary":       result_obj.raw_summary,
        "key_highlights":    result_obj.key_highlights,
        "guidance":          result_obj.guidance,
    }

    # ── Print extracted values ────────────────────────────────────────────────
    print("\n── Extracted values " + "─" * (W - 20))
    _DISPLAY_FIELDS = [
        ("period",            "Period"),
        ("revenue_cr",        "Revenue (Cr)"),
        ("pat_cr",            "PAT (Cr)"),
        ("ebitda_cr",         "EBITDA (Cr)"),
        ("ebitda_margin_pct", "EBITDA margin %"),
        ("eps",               "EPS"),
        ("confidence",        "Confidence"),
    ]
    for key, label in _DISPLAY_FIELDS:
        val = result.get(key)
        flag = ""
        if val is None:
            display = "— (missing)"
            flag = " ⚠"
        elif isinstance(val, float):
            display = f"{val:,.2f}"
        else:
            display = str(val)
        print(f"  {label:<22} {display}{flag}")

    summary = result.get("raw_summary", "")
    low_conf = summary.startswith("[LOW CONFIDENCE]")
    if low_conf:
        print(f"\n  ⚠  LOW CONFIDENCE flagged")
        summary = summary[len("[LOW CONFIDENCE]"):].strip()
    print(f"\n  Summary  : {summary}")

    hl = result.get("key_highlights") or []
    if hl:
        print("  Highlights:")
        for h in hl:
            print(f"    • {h}")

    guidance = result.get("guidance")
    if guidance:
        print(f"  Guidance : {guidance}")

    # ── Score against expected ────────────────────────────────────────────────
    if expected:
        fx_dummy = {"declared_unit": "unknown", "expected": expected}
        print("\n── Scores vs expected " + "─" * (W - 22))

        ns_list = score_numerics(result, expected, tol)
        for ns in ns_list:
            if ns.expected is None:
                continue
            mark = "✓" if ns.passed else "✗"
            got  = _fmt_num(ns.extracted)
            exp  = _fmt_num(ns.expected)
            pct  = f"  ({ns.pct_err*100:.1f}% err)" if ns.pct_err is not None else ""
            print(f"  {mark} {ns.key:<22} got={got:>10}  exp={exp:>10}{pct}")

        unit_ok, unit_detail = score_unit_conversion(result, fx_dummy)
        print(f"  {'✓' if unit_ok else '✗'} unit_conversion   {'' if unit_ok else unit_detail}")

        if expected.get("period"):
            pm, pm_detail = score_period(result, expected)
            print(f"  {'✓' if pm else '✗'} period_match      {'' if pm else pm_detail}")

        conf_ok, conf_exp, conf_got = score_confidence(result)
        print(f"  {'✓' if conf_ok else '✗'} confidence_calib  got='{conf_got}'  expected='{conf_exp}'")

    # ── Save fixture ──────────────────────────────────────────────────────────
    if save_fixture:
        period = result.get("period") or broadcast_dt[:10]
        fix_id = f"{symbol}_{period.replace(' ', '').replace('/', '')}"
        saved_expected = {k: result[k] for k in ("revenue_cr", "pat_cr", "eps", "ebitda_cr", "ebitda_margin_pct")
                         if result.get(k) is not None}
        if period:
            saved_expected["period"] = period
        if expected:
            saved_expected.update(expected)  # override with user-provided ground truth

        fixture = {
            "id":          fix_id,
            "symbol":      symbol,
            "company":     company,
            "period":      period,
            "period_type": "quarterly",
            "broadcast_dt": broadcast_dt[:10],
            "pdf_url":     pdf_source if pdf_source.startswith("http") else pdf_source,
            "declared_unit": "unknown",
            "expected":    saved_expected,
            "verified":    bool(expected),   # mark verified if user supplied expected values
            "notes":       "Added via --pdf debug run" + (" — user-verified" if expected else " — auto-extracted, verify before trusting"),
        }

        # Don't duplicate
        existing_ids: set[str] = set()
        if fixture_path.exists():
            for line in fixture_path.read_text(encoding="utf-8").splitlines():
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

        if fix_id in existing_ids:
            print(f"\n  Fixture '{fix_id}' already exists — not overwriting.")
        else:
            with fixture_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fixture) + "\n")
            print(f"\n  Saved fixture '{fix_id}' → {fixture_path}")

    print("\n" + "═" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval(
    fixtures: list[dict],
    tol: float = 0.02,
    use_langsmith: bool = True,
    force_vision: bool = False,
) -> list[FixtureScore]:
    scores: list[FixtureScore] = []

    for i, fx in enumerate(fixtures, 1):
        fix_id = fx.get("id", fx["symbol"])
        print(f"\n[{i}/{len(fixtures)}] {fix_id} — {fx.get('company','')}", flush=True)

        sc = FixtureScore(fixture_id=fix_id, symbol=fx["symbol"], period=fx.get("period", ""))
        result: dict | None = None

        try:
            t0 = time.perf_counter()
            result = _extract_fixture(fx, force_vision=force_vision)
            elapsed = time.perf_counter() - t0
            print(f"  extracted in {elapsed:.1f}s", flush=True)
        except Exception as e:
            sc.error = str(e)
            print(f"  FAILED: {e}", flush=True)
            scores.append(sc)
            continue

        if result is None:
            sc.error = "extractor returned None"
            scores.append(sc)
            continue

        expected = fx.get("expected", {})

        sc.numeric        = score_numerics(result, expected, tol)
        sc.unit_ok, _     = score_unit_conversion(result, fx)
        sc.period_match, _= score_period(result, expected)
        sc.confidence_ok, sc.confidence_exp, sc.confidence_got = score_confidence(result)

        if use_langsmith and os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1"):
            _langsmith_log(fx, result, sc)

        scores.append(sc)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial extraction eval")

    # ── Single-PDF debug mode ────────────────────────────────────────────────
    pdf_grp = parser.add_argument_group("Single-PDF debug mode")
    pdf_grp.add_argument("--pdf",              metavar="URL_OR_PATH", help="PDF URL or local path to debug")
    pdf_grp.add_argument("--symbol",           help="Ticker symbol (required with --pdf)")
    pdf_grp.add_argument("--company",          help="Company name (required with --pdf)")
    pdf_grp.add_argument("--broadcast-dt",     default="", help="Filing date YYYY-MM-DD (default: today)")
    pdf_grp.add_argument("--save-fixture",     action="store_true", help="Save result as a fixture entry")
    pdf_grp.add_argument("--force-vision",     action="store_true", help="Bypass rule-based + Ollama, use Gemini Flash directly")
    pdf_grp.add_argument("--expected-revenue", type=float, metavar="CR", help="Expected revenue in Crores")
    pdf_grp.add_argument("--expected-pat",     type=float, metavar="CR", help="Expected PAT in Crores")
    pdf_grp.add_argument("--expected-eps",     type=float,             help="Expected EPS")
    pdf_grp.add_argument("--expected-ebitda",  type=float, metavar="CR", help="Expected EBITDA in Crores")
    pdf_grp.add_argument("--expected-period",  metavar="STR",          help='Expected period e.g. "Q1 FY2027"')

    # ── Fixture suite mode ───────────────────────────────────────────────────
    suite_grp = parser.add_argument_group("Fixture suite mode")
    suite_grp.add_argument("--fixture",        default=str(DEFAULT_FIXTURES), help="Path to JSONL fixture file")
    suite_grp.add_argument("--verified-only",  action="store_true", help="Only run fixtures with verified=true")
    suite_grp.add_argument("--seed-from-db",   type=int, metavar="N", help="Seed N fixtures from DB then exit")
    suite_grp.add_argument("--no-langsmith",   action="store_true", help="Skip LangSmith logging")
    suite_grp.add_argument("--vision",         action="store_true", help="Force Gemini Flash vision path for all fixtures")
    suite_grp.add_argument("--tol",            type=float, default=0.02, help="Numeric tolerance (default 0.02 = 2%%)")

    args = parser.parse_args()

    cfg          = AppConfig.from_env()
    fixture_path = Path(args.fixture)

    # ── Seed mode ─────────────────────────────────────────────────────────────
    if args.seed_from_db:
        seed_from_db(cfg.db_path, args.seed_from_db, fixture_path)
        return

    # ── Single-PDF debug mode ─────────────────────────────────────────────────
    if args.pdf:
        if not args.symbol or not args.company:
            parser.error("--pdf requires --symbol and --company")

        expected: dict = {}
        if args.expected_revenue is not None: expected["revenue_cr"]  = args.expected_revenue
        if args.expected_pat     is not None: expected["pat_cr"]      = args.expected_pat
        if args.expected_eps     is not None: expected["eps"]         = args.expected_eps
        if args.expected_ebitda  is not None: expected["ebitda_cr"]   = args.expected_ebitda
        if args.expected_period  is not None: expected["period"]      = args.expected_period

        run_single_pdf(
            pdf_source=args.pdf,
            symbol=args.symbol,
            company=args.company,
            broadcast_dt=args.broadcast_dt,
            expected=expected or None,
            tol=args.tol,
            save_fixture=args.save_fixture,
            fixture_path=fixture_path,
            force_vision=args.force_vision,
        )
        return

    # ── Fixture suite mode ────────────────────────────────────────────────────
    if not fixture_path.exists():
        print(f"Fixture file not found: {fixture_path}")
        print("Run with --seed-from-db N to generate fixtures from the DB.")
        sys.exit(1)

    fixtures: list[dict] = []
    for line in fixture_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            fx = json.loads(line)
            if args.verified_only and not fx.get("verified", False):
                print(f"  SKIP (unverified): {fx.get('id', fx['symbol'])}")
                continue
            fixtures.append(fx)
        except json.JSONDecodeError as e:
            print(f"  Bad fixture line: {e}")

    if not fixtures:
        print("No fixtures to run.")
        sys.exit(0)

    force_vision = getattr(args, "vision", False)
    pipeline_label = "Gemini Flash vision" if force_vision else "full pipeline"
    print(f"Running extraction eval on {len(fixtures)} fixtures via {pipeline_label} (tol={args.tol*100:.0f}%)…")
    print("Each fixture downloads + re-extracts its PDF — expect ~15-60s per fixture.\n")

    scores = run_eval(fixtures, tol=args.tol, use_langsmith=not args.no_langsmith, force_vision=force_vision)
    print_report(scores)


if __name__ == "__main__":
    main()
