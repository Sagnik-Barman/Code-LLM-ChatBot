"""
Stage 1 data prep: builds a continued-pretraining corpus of Python + C++ code.

Clones two well-known, MIT-licensed reference repos of algorithm and data-
structure implementations -- covering arrays, strings, graphs, trees, DP,
sorting, searching, etc., i.e. the same topic spread called for in the
training plan -- and concatenates their source into one text file that
train.py can pretrain on directly.

Usage:
    python prepare_code_corpus.py
    python prepare_code_corpus.py --out data/code_corpus.txt --max_mb 50
    python prepare_code_corpus.py --languages python        # skip C++
"""
import argparse
import os
import subprocess
import tempfile

REPOS = {
    "python": ("https://github.com/TheAlgorithms/Python.git", (".py",)),
    "cpp": ("https://github.com/TheAlgorithms/C-Plus-Plus.git", (".cpp", ".hpp", ".h")),
}
SKIP_DIR_NAMES = {".git", "test", "tests", "__pycache__", ".github"}


def clone_repo(url, dest):
    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True, capture_output=True)


def collect_source_files(root, extensions):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fname in filenames:
            if fname.endswith(extensions):
                files.append(os.path.join(dirpath, fname))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/code_corpus.txt")
    parser.add_argument("--max_mb", type=float, default=50.0,
                         help="Stop appending once the corpus reaches roughly this size")
    parser.add_argument("--languages", nargs="+", choices=list(REPOS.keys()),
                         default=list(REPOS.keys()))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    max_bytes = int(args.max_mb * 1024 * 1024)
    written_bytes, file_count = 0, 0

    with tempfile.TemporaryDirectory() as tmp, open(args.out, "w", encoding="utf-8") as out_f:
        for lang in args.languages:
            url, extensions = REPOS[lang]
            repo_name = url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "")
            dest = os.path.join(tmp, repo_name)
            print(f"Cloning {url} ...")
            clone_repo(url, dest)

            source_files = collect_source_files(dest, extensions)
            print(f"  {len(source_files)} source files found in {repo_name}")

            for path in source_files:
                if written_bytes >= max_bytes:
                    print(f"  Reached --max_mb limit ({args.max_mb} MB), stopping early.")
                    break
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                if not content.strip():
                    continue
                # File-boundary marker: helps the model learn "a new file
                # starts here" instead of blending unrelated files together.
                chunk = f"\n# --- file: {os.path.relpath(path, dest)} ---\n{content}\n"
                out_f.write(chunk)
                written_bytes += len(chunk.encode("utf-8"))
                file_count += 1

    print(f"\nWrote {file_count} files ({written_bytes/1024/1024:.1f} MB) to {args.out}")
    print("Next: continued pretraining --")
    print(f"  python train.py --data {args.out} --epochs 3 --out checkpoints/code_pretrained.pth")
    print("Then point instruction fine-tuning at that checkpoint's weights before running Stage 2.")


if __name__ == "__main__":
    main()
