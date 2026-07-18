"""
preflight.py — verify EVERYTHING before spending a single API dollar.

    python preflight.py            # all checks except live API calls
    python preflight.py --live     # also makes one tiny call per provider (~$0.001)

Checks, in dependency order:
  1. Python version           (3.10/3.11 required for wheel availability)
  2. Required packages        (importable, with versions)
  3. PyTorch + CUDA           (GPU visible = fast embedding; CPU = OK, slower)
  4. tiktoken encoding        (downloads ~2 MB on first use)
  5. Harness modules          (all 9 import cleanly; serialization round-trip)
  6. API keys present         (env vars set, plausible format)
  7. API reachability         (--live only: 1-token call to each provider)
  8. Dataset files            (data/corpus.jsonl 50 docs, questions.jsonl 25)
  9. Git available            (needed for the evidence commit hash)
 10. Disk space               (>= 6 GB free recommended)

Exit code 0 = green light. Any FAIL prints exactly what to do.
"""

import importlib
import os
import shutil
import subprocess
import sys

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
results = []  # (status, name, detail)


def report(ok, name, detail="", warn=False):
    status = "WARN" if (warn and not ok) else ("PASS" if ok else "FAIL")
    color = GREEN if status == "PASS" else (YELLOW if status == "WARN" else RED)
    print(f"  [{color}{status}{RESET}] {name}" + (f" — {detail}" if detail else ""))
    results.append((status, name, detail))
    return ok


def main():
    live = "--live" in sys.argv
    print("\n=== PREFLIGHT — thesis harness ===\n")

    # 1 — Python version -------------------------------------------------
    v = sys.version_info
    report(v[:2] in [(3, 10), (3, 11)], "Python version",
           f"{v.major}.{v.minor}.{v.micro} "
           + ("" if v[:2] in [(3, 10), (3, 11)] else
              "-> install 3.11.9 and recreate the venv with: py -3.11 -m venv .venv"))

    # 2 — Required packages ----------------------------------------------
    packages = {
        "numpy": "numpy", "openai": "openai", "anthropic": "anthropic",
        "tiktoken": "tiktoken", "yaml": "pyyaml", "datasets": "datasets",
        "sentence_transformers": "sentence-transformers", "faiss": "faiss-cpu",
        "torch": "torch",
    }
    missing = []
    for mod, pipname in packages.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            report(True, f"package {pipname}", ver)
        except Exception as e:
            missing.append(pipname)
            report(False, f"package {pipname}", f"{type(e).__name__}: {e}")
    if missing:
        print(f"\n  Fix: pip install {' '.join(missing)}\n")

    # 3 — PyTorch + CUDA ---------------------------------------------------
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "none"
        report(True, "CUDA GPU", f"{name}" if cuda else
               "not available — embedding runs on CPU (works, ~10x slower). "
               "For the 4060 Ti: pip install torch --index-url "
               "https://download.pytorch.org/whl/cu121", warn=not cuda)
    except Exception as e:
        report(False, "CUDA check", str(e))

    # 4 — tiktoken encoding (needs one-time network fetch) -----------------
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        report(enc.encode("test") is not None, "tiktoken encoding", "cl100k_base ready")
    except Exception as e:
        report(False, "tiktoken encoding",
               f"{e} -> check internet/firewall; it fetches ~2 MB once")

    # 5 — Harness modules + serialization sanity ---------------------------
    try:
        from config import SCOPES, cell_id                      # noqa
        from chunking import chunk                              # noqa
        from serialise import serialize, roundtrip_ok
        from tokens import count_gpt                            # noqa
        import contextualise, embed_index, generate, evaluate   # noqa
        rec = {"doc_id": "d0", "title": "T, \"q\"", "chunk_id": "d0_c00",
               "context": "Line.\nTwo.", "text": "Body: text."}
        sizes = {f: len(serialize(rec, f)) for f in ("json", "toon", "yaml")}
        ok = roundtrip_ok(rec) and sizes["toon"] < sizes["json"]
        report(ok, "harness modules",
               f"round-trip OK; chars json={sizes['json']} "
               f"yaml={sizes['yaml']} toon={sizes['toon']}")
        report(len(SCOPES["pilot"]) == 2 and len(SCOPES["core"]) == 18,
               "scopes", "pilot=2 cells, core=18 cells")
    except Exception as e:
        report(False, "harness modules",
               f"{type(e).__name__}: {e} -> run preflight from the repo root")

    # 6 — API keys -----------------------------------------------------------
    ok_o = os.environ.get("OPENAI_API_KEY", "").startswith("sk-")
    ok_a = os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-")
    report(ok_o, "OPENAI_API_KEY",
           "set" if ok_o else 'missing -> $env:OPENAI_API_KEY = "sk-..."')
    report(ok_a, "ANTHROPIC_API_KEY",
           "set" if ok_a else 'missing -> $env:ANTHROPIC_API_KEY = "sk-ant-..."')

    # 7 — Live API reachability (opt-in, ~ $0.001 total) ----------------------
    if live and ok_o:
        try:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model="gpt-4o-mini", max_tokens=1,
                messages=[{"role": "user", "content": "hi"}])
            report(bool(r.choices), "OpenAI live call", "auth + credit OK")
        except Exception as e:
            report(False, "OpenAI live call",
                   f"{e} -> check billing/credit at platform.openai.com")
    if live and ok_a:
        try:
            from anthropic import Anthropic
            r = Anthropic().messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1,
                messages=[{"role": "user", "content": "hi"}])
            report(bool(r.content is not None), "Anthropic live call",
                   "auth + credit OK")
        except Exception as e:
            report(False, "Anthropic live call",
                   f"{e} -> check billing/credit at console.anthropic.com")
    if not live:
        report(False, "live API calls", "skipped — rerun with --live to test "
               "auth + credit (~$0.001)", warn=True)

    # 8 — Dataset files ---------------------------------------------------------
    for fname, expected in [("data/corpus.jsonl", 50), ("data/questions.jsonl", 25)]:
        if os.path.exists(fname):
            n = sum(1 for _ in open(fname, encoding="utf-8"))
            report(n == expected, fname, f"{n} lines (expected {expected})")
        else:
            report(False, fname,
                   "missing -> run: python prepare_hotpotqa.py", warn=True)

    # 9 — Git ----------------------------------------------------------------------
    git = shutil.which("git")
    if git:
        try:
            h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
            report(True, "git", f"repo initialized, HEAD {h}")
        except Exception:
            report(True, "git", "installed; repo not initialized yet "
                   "(git init before the evidence commit)", warn=True)
    else:
        report(False, "git", "not installed -> winget install --id Git.Git -e")

    # 10 — Disk space -----------------------------------------------------------------
    free_gb = shutil.disk_usage(".").free / 1e9
    report(free_gb >= 6, "free disk", f"{free_gb:.1f} GB "
           + ("" if free_gb >= 6 else "-> free at least 6 GB"))

    # Verdict ----------------------------------------------------------------------------
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print("\n" + "=" * 50)
    if not fails:
        print(f"{GREEN}GREEN LIGHT{RESET} — {len(results)} checks, "
              f"{len(warns)} warnings. Ready:  python run.py --scope pilot")
    else:
        print(f"{RED}{len(fails)} BLOCKER(S){RESET}:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        print("Fix the items above, then rerun:  python preflight.py --live")
    print("=" * 50)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
