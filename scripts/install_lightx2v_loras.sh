#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STORE_ROOT="${H3_MODEL_STORE:-/root/h3-model-store}"
STORE_DIR="$STORE_ROOT/loras/lightx2v"
LINK_DIR="$PROJECT_DIR/models/loras/lightx2v"
REPOSITORY="lightx2v/Minimax-h3-Turbo"

mkdir -p "$STORE_DIR" "$LINK_DIR"

install_weight() {
  local filename="$1"
  local expected_sha256="$2"
  local destination="$STORE_DIR/$filename"
  local temporary="$destination.partial"

  if [[ -f "$destination" ]] && printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum --check --status; then
    echo "Already verified: $filename"
  else
    rm -f "$temporary"
    curl --fail --location --retry 5 --retry-delay 2 \
      "https://huggingface.co/$REPOSITORY/resolve/main/$filename?download=true" \
      --output "$temporary"
    printf '%s  %s\n' "$expected_sha256" "$temporary" | sha256sum --check --status
    mv "$temporary" "$destination"
    echo "Installed: $filename"
  fi
  ln -sfn "$destination" "$LINK_DIR/$filename"
}

install_weight \
  "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors" \
  "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4"
install_weight \
  "minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors" \
  "b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966"
install_weight \
  "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors" \
  "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99"

echo "LightX2V LoRA installation ready under $LINK_DIR"
