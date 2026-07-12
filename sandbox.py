"""
Sandboxed code execution, used as the automated correctness gate for the
autonomous trainer. This is what stands in for a human reviewer: instead of
someone approving each user interaction before it's trained on, code is only
kept for training if it actually runs (Python) or compiles+runs (C++).

IMPORTANT SCOPE NOTE: this is a *correctness* filter (subprocess + timeout +
memory limit), not a hardened security sandbox. It's appropriate for filtering
your own chatbot's logged interactions on a machine you control. It is NOT
safe to point at arbitrary/adversarial untrusted code on shared or
multi-tenant infrastructure -- for that you'd want real isolation (a
container, gVisor, firecracker VM, etc.).
"""
import os
import re
import subprocess
import sys
import tempfile

try:
    import resource  # POSIX only; unavailable on Windows
except ImportError:
    resource = None

PYTHON_TIMEOUT = 5
CPP_RUN_TIMEOUT = 5
CPP_COMPILE_TIMEOUT = 15
MAX_MEMORY_MB = 256

_CODE_BLOCK_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)


def _limit_resources():
    """Applied in the child process via preexec_fn (POSIX only)."""
    if resource is None:
        return
    mem_bytes = MAX_MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (PYTHON_TIMEOUT, PYTHON_TIMEOUT))


def extract_code_blocks(text, language=None):
    """
    Pull fenced ```lang ... ``` code blocks out of a response.
    `language`, if given, is a set of lowercase language tags to keep
    (e.g. {"python", "py"}); None keeps everything.

    Falls back to treating the WHOLE response as one unfenced code candidate
    if no fences are found -- models fine-tuned on CodeAlpaca-style data
    typically emit raw code with no markdown fences at all, so requiring
    fences would silently discard every real answer.
    """
    blocks = []
    for lang, code in _CODE_BLOCK_RE.findall(text):
        lang = (lang or "").lower()
        if language is None or lang in language:
            blocks.append((lang, code.strip()))

    if not blocks and text.strip():
        looks_cpp = bool(re.search(r"#include\s*<|std::|int\s+main\s*\(", text))
        guessed_lang = "cpp" if looks_cpp else "python"
        if language is None or guessed_lang in language:
            blocks.append((guessed_lang, text.strip()))

    return blocks


def run_python(code, timeout=PYTHON_TIMEOUT):
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limit_resources if os.name != "nt" else None,
        )
        return proc.returncode == 0, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    finally:
        os.unlink(path)


def run_cpp(code, timeout=CPP_RUN_TIMEOUT):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "main.cpp")
        binp = os.path.join(d, "main.out")
        with open(src, "w") as f:
            f.write(code)
        try:
            compile_proc = subprocess.run(
                ["g++", "-O2", "-std=c++17", src, "-o", binp],
                capture_output=True, text=True, timeout=CPP_COMPILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, "", "compile timeout"
        if compile_proc.returncode != 0:
            return False, "", compile_proc.stderr
        try:
            run_proc = subprocess.run([binp], capture_output=True, text=True, timeout=timeout)
            return run_proc.returncode == 0, run_proc.stdout, run_proc.stderr
        except subprocess.TimeoutExpired:
            return False, "", "run timeout"


def passes_execution_filter(response_text):
    """
    True if at least one code block in the response actually runs/compiles
    cleanly. A response with no code, or only broken code, is rejected --
    this is the safety gate that makes autonomous (no-human) training viable.
    """
    blocks = extract_code_blocks(response_text, language={"python", "py", "cpp", "c++"})
    if not blocks:
        return False
    for lang, code in blocks:
        if lang in ("python", "py"):
            ok, _, _ = run_python(code)
        elif lang in ("cpp", "c++"):
            ok, _, _ = run_cpp(code)
        else:
            continue
        if ok:
            return True
    return False
