#!/usr/bin/env bash
# Download the ten CC BY 4.0 papers used by eval/ and the demo (not redistributed here).
set -euo pipefail
cd "$(dirname "$0")/data" 2>/dev/null || { mkdir -p "$(dirname "$0")/data"; cd "$(dirname "$0")/data"; }; mkdir -p papers
for id in 2310.06825 2402.00838 2404.14219 2408.00118 2501.00656 2502.02737 2309.16609 2401.04088 2403.08295 2406.11704; do
  [ -f "papers/$id.pdf" ] || curl -sSL -o "papers/$id.pdf" "https://arxiv.org/pdf/$id"
done
# The evaluation was run on these exact files; arXiv serves the latest version, so a
# mismatch means the paper was revised since.
while read -r id want; do
  got=$(sha256sum "papers/$id.pdf" | cut -c1-16)
  [ "$got" = "$want" ] || echo "warning: papers/$id.pdf differs from the evaluated copy ($got, expected $want)"
done < ../eval/papers.sha256
