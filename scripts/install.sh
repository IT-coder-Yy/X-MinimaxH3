#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${H3_SERVE_RUNTIME_DIR:-${release_root}/runtime}"
model_root="${H3_SERVE_MODEL_DIR:-${release_root}/models}"
profile="full"
model_source="auto"
skip_models=0
skip_system=0
accept_license=0
dry_run=0
repair_models=0

usage() {
  cat <<'EOF'
X-MinimaxH3 一键安装

用法：./install.sh [选项]
  --profile full|core|fl2va|ref2va|upscaler  默认 full
  --source auto|modelscope|hf-mirror|huggingface  默认 auto
  --without-upscaler      等价于 full -> core，不安装 FlashVSR
  --accept-model-license  确认接受 MiniMax H3/LoRA/FlashVSR 模型许可
  --repair-models         覆盖清单路径中校验失败的模型文件
  --skip-models           只安装运行环境，不下载权重
  --skip-system-packages  不调用 apt，只使用已有 Python/ffmpeg
  --dry-run               只显示计划，不修改系统
EOF
}

while (($#)); do
  case "$1" in
    --profile) profile="${2:?--profile needs a value}"; shift 2 ;;
    --source) model_source="${2:?--source needs a value}"; shift 2 ;;
    --without-upscaler) [[ "${profile}" == "full" ]] && profile="core"; shift ;;
    --accept-model-license) accept_license=1; shift ;;
    --repair-models) repair_models=1; shift ;;
    --skip-models) skip_models=1; shift ;;
    --skip-system-packages) skip_system=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${profile}" in full|core|fl2va|ref2va|upscaler) ;; *) echo "无效 profile：${profile}" >&2; exit 2;; esac
case "${model_source}" in auto|modelscope|hf-mirror|huggingface) ;; *) echo "无效 source：${model_source}" >&2; exit 2;; esac

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  echo "运行时只支持 Linux x86_64；Windows 请运行 install-windows.ps1（WSL2）。" >&2
  exit 1
}

need_upscaler=0
[[ "${profile}" == "full" || "${profile}" == "upscaler" ]] && need_upscaler=1

echo "X-MinimaxH3 安装计划"
echo "  目录：${release_root}"
echo "  模型：${profile}"
echo "  下载源：${model_source}（auto 优先 ModelScope）"
echo "  FlashVSR：$([[ ${need_upscaler} == 1 ]] && echo 是 || echo 否)"
echo "  模型许可：MiniMax H3 Community License（见 third_party_licenses/）"
echo "             该协议含地域限制：不授权在欧盟、英国、韩国和美国使用或分发。"
if [[ "${dry_run}" == 1 ]]; then
  echo "  dry-run：不会安装或下载"
  exit 0
fi

if [[ "${skip_models}" == 0 && "${accept_license}" == 0 ]]; then
  if [[ -t 0 ]]; then
    echo
    echo "模型权重及随包MiniMax运行材料受各自许可证约束；详情见 THIRD_PARTY_NOTICES.md。"
    echo "继续表示你已经阅读本地协议全文，并确认自己的地域和用途获得许可。"
    read -r -p "确认已经阅读并接受这些许可证？[y/N] " answer
    [[ "${answer}" =~ ^[Yy]$ ]] || { echo "已取消。"; exit 1; }
    accept_license=1
  else
    echo "非交互安装必须传入 --accept-model-license。" >&2
    exit 2
  fi
fi

install_ubuntu_packages() {
  local apt_cmd=(apt-get)
  if [[ "$(id -u)" != 0 ]]; then
    command -v sudo >/dev/null || {
      echo "需要 sudo 安装系统依赖，或使用 --skip-system-packages。" >&2
      return 1
    }
    apt_cmd=(sudo apt-get)
  fi
  "${apt_cmd[@]}" update
  "${apt_cmd[@]}" install -y ca-certificates curl ffmpeg git software-properties-common \
    python3.10 python3.10-dev python3.10-venv
  if [[ "${need_upscaler}" == 1 ]] && ! command -v python3.11 >/dev/null; then
    local add_repo=(add-apt-repository)
    [[ "$(id -u)" != 0 ]] && add_repo=(sudo add-apt-repository)
    "${add_repo[@]}" -y ppa:deadsnakes/ppa
    "${apt_cmd[@]}" update
    "${apt_cmd[@]}" install -y python3.11 python3.11-dev python3.11-venv
  fi
}

if [[ "${skip_system}" == 0 ]]; then
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
  fi
  if [[ "${ID:-}" == "ubuntu" ]]; then
    install_ubuntu_packages
  else
    echo "非 Ubuntu 系统：不自动调用包管理器，将检查现有依赖。"
  fi
fi

command -v python3.10 >/dev/null || {
  echo "缺少 Python 3.10（需要 venv 模块）。" >&2; exit 1;
}
command -v ffmpeg >/dev/null || {
  echo "缺少 ffmpeg。" >&2; exit 1;
}
if [[ "${need_upscaler}" == 1 ]]; then
  command -v python3.11 >/dev/null || {
    echo "完整安装需要 Python 3.11；或改用 --without-upscaler。" >&2; exit 1;
  }
fi

mkdir -p "${runtime_root}" "${model_root}" "${release_root}/output" \
  "${release_root}/data" "${release_root}/workspace/default"

pip_index="${H3_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
main_venv="${runtime_root}/venv"
python3.10 -m venv "${main_venv}"
python="${main_venv}/bin/python"
"${python}" -m pip install --index-url "${pip_index}" --upgrade pip setuptools wheel
"${python}" -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126
"${python}" -m pip install --index-url "${pip_index}" -r "${release_root}/requirements.lock"
"${python}" -m pip install --no-deps \
  "${release_root}/wheels/sageattention-2.2.0-cp310-cp310-linux_x86_64.whl"
"${python}" -m pip freeze > "${runtime_root}/installed-packages.txt"

if [[ "${need_upscaler}" == 1 ]]; then
  flash_venv="${runtime_root}/flashvsr-venv"
  python3.11 -m venv "${flash_venv}"
  flash_python="${flash_venv}/bin/python"
  "${flash_python}" -m pip install --index-url "${pip_index}" --upgrade pip setuptools wheel packaging ninja
  "${flash_python}" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
  "${flash_python}" -m pip install --index-url "${pip_index}" -r "${release_root}/requirements-flashvsr.lock"
  "${flash_python}" -m pip install --no-deps \
    "${release_root}/wheels/block_sparse_attn-0.0.2-cp311-cp311-linux_x86_64.whl"
  "${flash_python}" -m pip freeze > "${runtime_root}/flashvsr-installed-packages.txt"
fi

if [[ "${skip_models}" == 0 ]]; then
  download_args=(
    "${release_root}/scripts/download_models.py"
    "${model_root}"
    --profile "${profile}"
    --source "${model_source}"
    --accept-model-license
  )
  [[ "${repair_models}" == 1 ]] && download_args+=(--repair)
  "${python}" "${download_args[@]}"
fi

chmod +x "${release_root}"/*.sh "${release_root}/scripts"/*.sh
doctor_args=("${release_root}/scripts/doctor.py" --profile "${profile}")
[[ "${skip_models}" == 1 ]] && doctor_args+=(--skip-models)
echo "正在执行安装后运行时自检…"
H3_SERVE_RUNTIME_DIR="${runtime_root}" H3_SERVE_MODEL_DIR="${model_root}" \
  "${python}" "${doctor_args[@]}"
echo
echo "安装完成。"
echo "  自检：./scripts/doctor.py --profile ${profile}"
echo "  启动：./start.sh"
echo "  控制台：http://127.0.0.1:8090"
