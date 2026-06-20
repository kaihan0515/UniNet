#!/usr/bin/env bash
# 把 RESULTS_REPORT.md 轉成 PDF（pandoc 內嵌圖 -> HTML -> Edge headless 列印）。
# 從 repo 根目錄執行：bash report/build_pdf.sh
cd "$(dirname "$0")/.." || exit 1

pandoc RESULTS_REPORT.md -o RESULTS_REPORT.html --standalone --self-contained \
  --css report/report.css --metadata title="鋼珠表面瑕疵檢測 — 成果報告"

EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
[ -x "$EDGE" ] || EDGE="/c/Program Files/Google/Chrome/Application/chrome.exe"
"$EDGE" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(pwd -W 2>/dev/null || pwd)/RESULTS_REPORT.pdf" \
  "file:///$(pwd -W 2>/dev/null || pwd)/RESULTS_REPORT.html"

echo "done -> RESULTS_REPORT.pdf"
