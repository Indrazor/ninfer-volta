#!/usr/bin/env bash
# Scaffold the v100 target tree: mirror every per-target source directory
# (rtx5090/, dgx_spark/) with placeholder v100/ files so CMake can resolve
# NINFER_TARGET-templated paths at configure/generate time. Each placeholder
# is replaced by the real Volta kernel as its milestone in PORTING.md lands.
# Re-runnable; never overwrites existing files.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

created=0
while IFS= read -r f; do
  new="${f/\/rtx5090\//\/v100\/}"
  new="${new/\/dgx_spark\//\/v100\/}"
  [ -e "$new" ] && continue
  mkdir -p "$(dirname "$new")"
  case "$new" in
    */sources.cmake)
      printf '# Volta (sm_70): pending port, see PORTING.md\n' > "$new" ;;
    *.cu|*.cpp|*.c|*.h|*.cuh|*.hpp)
      printf '// Volta (sm_70): pending port placeholder, see PORTING.md\n' > "$new" ;;
    *)
      printf '# placeholder, see PORTING.md\n' > "$new" ;;
  esac
  created=$((created+1))
done < <(find src third_party -type f \( -path '*/rtx5090/*' -o -path '*/dgx_spark/*' \) | sort)

echo "created $created v100 placeholder files"
