#!/usr/bin/env bash
# 编译 BidKV SC 2026 论文
# 用法：cd paper && bash build.sh
#       或从仓库根目录：bash paper/build.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN="bidkv_sc2026"

echo "[1/4] pdflatex (first pass)..."
pdflatex -interaction=nonstopmode "$MAIN.tex"

echo "[2/4] bibtex (references)..."
bibtex "$MAIN"

echo "[3/4] pdflatex (second pass)..."
pdflatex -interaction=nonstopmode "$MAIN.tex"

echo "[4/4] pdflatex (third pass, cross-references)..."
pdflatex -interaction=nonstopmode "$MAIN.tex"

echo ""
echo "Done: $SCRIPT_DIR/$MAIN.pdf"
