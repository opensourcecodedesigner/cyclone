#!/usr/bin/env python3
"""
AEGIS Pipeline Fallback Verifier
=================================
Runs the backtest (or reads a saved log) and checks for the EXACT fallback
signatures identified in the QA diagnostic report. Prints a pass/fail table
instead of trusting narrative summaries.

Usage:
    python verify_aegis_fix.py --run                # runs backtest_fani.py live
    python verify_aegis_fix.py --log path/to/log.txt  # checks an existing log
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# (label, regex pattern, meaning if FOUND)
FALLBACK_SIGNATURES = [
    ("julia_offline",
     r"Local Julia server offline|ConnectionRefusedError|Max retries exceeded.*8080",
     "Julia physics server unreachable -> hardcoded fallback dict in use"),

    ("julia_hardcoded_latency",
     r"elapsed_ms\s*=\s*1940\.5|Physics Latency\s*:\s*1940\.5",
     "The exact old hardcoded constant (1940.5ms) reappeared verbatim"),

    ("gemini_key_missing",
     r"No API key found for (Gemini|Google)|GEMINI_API_KEY.*None",
     "Gemini key not loaded at runtime"),

    ("jev_proxy_triage",
     r"Jev Proxy",
     "System 1 triage ran on local deterministic proxy, not Gemini"),

    ("deterministic_cap",
     r"Deterministic CAP Synthesis|Local execution mode",
     "CAP dispatch ran on hardcoded string template, not Gemini"),

    ("openai_key_missing_rag",
     r"No API key found for OpenAI|Could not load OpenAI embedding model",
     "LlamaIndex fell back to direct file read instead of vector search"),

    ("iou_hardcoded_ratio",
     r"0\.812|min\(A\)\s*x\s*0\.812",
     "IoU still using the old approximated constant, not real Shapely geometry"),
]

# Signals that a subsystem IS genuinely live (positive confirmation, not just
# absence of the fallback string above)
POSITIVE_SIGNATURES = [
    ("julia_health_ok",     r"Listening on 0\.0\.0\.0:8080|Julia Physics Microservice Succeeded"),
    ("gemini_call_made",    r"gemini-|models/gemini|genai\.Client|HTTP/1\.1\" 200.*generativelanguage"),
    ("shapely_geometry",    r"shapely|intersection\(.*union\(|real geometric intersection"),
    ("dotenv_loaded",       r"load_dotenv\(\)|Loaded \.env"),
]


def run_backtest(cwd: str) -> str:
    print(f"[*] Running backtest_fani.py in {cwd} ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "backtest_fani.py"],
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    dt = time.time() - t0
    print(f"[*] Finished in {dt:.1f}s (exit code {proc.returncode})")
    return proc.stdout + "\n" + proc.stderr


def load_log(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def analyze(text: str):
    print("\n" + "=" * 70)
    print("FALLBACK SIGNATURE CHECK  (found = still broken)")
    print("=" * 70)
    any_fail = False
    for label, pattern, meaning in FALLBACK_SIGNATURES:
        m = re.search(pattern, text, re.IGNORECASE)
        status = "FAIL (fallback detected)" if m else "clear"
        if m:
            any_fail = True
            snippet = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        print(f"[{'FAIL' if m else 'ok  '}] {label:28s} {status}")
        if m:
            print(f"       -> {meaning}")
            print(f"       -> context: ...{snippet}...")

    print("\n" + "=" * 70)
    print("POSITIVE LIVE-EXECUTION CHECK  (found = good sign)")
    print("=" * 70)
    for label, pattern in POSITIVE_SIGNATURES:
        m = re.search(pattern, text, re.IGNORECASE)
        print(f"[{'YES ' if m else 'no  '}] {label:28s} {'present' if m else 'not found'}")

    # cross-run consistency check: same asset depth appearing twice with
    # different values (the smoking gun from the last report)
    print("\n" + "=" * 70)
    print("DEPTH-VALUE CONSISTENCY CHECK")
    print("=" * 70)
    depths = re.findall(r"([a-z_]+)\s*:\s*[A-Z_]+\s*\([\d.]+%\s*Payout\s*@\s*([\d.]+)m\)", text)
    seen = {}
    inconsistent = False
    for asset, depth in depths:
        seen.setdefault(asset, set()).add(depth)
    for asset, vals in seen.items():
        if len(vals) > 1:
            inconsistent = True
            print(f"[FAIL] {asset}: inconsistent depths across stages -> {sorted(vals)}")
    if not inconsistent:
        print("[ok  ] all repeated asset depths matched across stages")

    print("\n" + "=" * 70)
    if any_fail or inconsistent:
        print("VERDICT: NOT FIXED — one or more fallback signatures still present.")
    else:
        print("VERDICT: No known fallback signatures detected. Still confirm "
              "manually that a real network call hit Gemini (check request logs "
              "/ billing / API dashboard), since text absence isn't 100% proof.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Run backtest_fani.py live")
    ap.add_argument("--cwd", default=r"d:\julia engine", help="Directory containing backtest_fani.py")
    ap.add_argument("--log", help="Path to an existing log file to analyze instead of running")
    args = ap.parse_args()

    if args.log:
        text = load_log(args.log)
    elif args.run:
        text = run_backtest(args.cwd)
    else:
        print("Specify --run or --log <path>. See --help.")
        sys.exit(1)

    analyze(text)


if __name__ == "__main__":
    main()
