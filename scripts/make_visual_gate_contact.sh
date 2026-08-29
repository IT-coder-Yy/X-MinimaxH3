#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 INPUT.mp4 OUTPUT.jpg 't0,t1,...'" >&2
  exit 2
fi

input=$1
output=$2
times_csv=$3
IFS=',' read -r -a times <<<"${times_csv}"
if [[ ${#times[@]} -lt 4 || ${#times[@]} -gt 8 ]]; then
  echo "visual gate requires 4-8 sample times" >&2
  exit 2
fi

filter=""
for index in "${!times[@]}"; do
  filter+="[0:v]trim=start=${times[$index]}:duration=0.042,setpts=PTS-STARTPTS,scale=480:-2[v${index}];"
done
for index in "${!times[@]}"; do
  filter+="[v${index}]"
done
if [[ ${#times[@]} -le 4 ]]; then
  columns=2
else
  columns=$(( (${#times[@]} + 1) / 2 ))
fi
layout=""
for index in "${!times[@]}"; do
  column=$(( index % columns ))
  row=$(( index / columns ))
  x="0"
  for ((part = 0; part < column; part++)); do
    if [[ ${part} -eq 0 ]]; then x="w0"; else x+="+w${part}"; fi
  done
  y="0"
  if [[ ${row} -gt 0 ]]; then y="h0"; fi
  if [[ -n ${layout} ]]; then layout+="|"; fi
  layout+="${x}_${y}"
done
filter+="xstack=inputs=${#times[@]}:layout=${layout}:fill=black[out]"

mkdir -p "$(dirname "${output}")"
ffmpeg -hide_banner -loglevel error -y -i "${input}" \
  -filter_complex "${filter}" -map '[out]' -frames:v 1 "${output}"
echo "${output}"
