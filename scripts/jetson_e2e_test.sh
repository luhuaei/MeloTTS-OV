#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REMOTE="${REMOTE:-nvidia@192.168.1.230}"
REMOTE_DIR="${REMOTE_DIR:-/home/nvidia/MeloTTS-OV}"
CONTAINER_NAME="${CONTAINER_NAME:-melotts-jetson-e2e}"
MELOTTS="${MELOTTS:-0.1.2}"
IMAGE_REPO="${IMAGE_REPO:-registry.lazycat.cloud/x/lzc-aipod-melotts-ov}"
IMAGE="${IMAGE:-${IMAGE_REPO}:${MELOTTS}}"
DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS:---runtime nvidia --network host}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
OFFLINE_ENV_ARGS="${OFFLINE_ENV_ARGS:--e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e NLTK_DATA=/root/nltk_data}"
SERVICE_PORT="${SERVICE_PORT:-8000}"
HEALTH_PATH="${HEALTH_PATH:-/healthz}"
SPEECH_PATH="${SPEECH_PATH:-/v1/audio/speech}"
REQUEST_TEXT="${REQUEST_TEXT:-你好，欢迎使用 MeloTTS 自动化测试。}"
REQUEST_VOICE="${REQUEST_VOICE:-ZH}"
REQUEST_FORMAT="${REQUEST_FORMAT:-wav}"
REQUEST_SPEED="${REQUEST_SPEED:-1.0}"
MIXED_LANG_TEST="${MIXED_LANG_TEST:-1}"
MIXED_REQUEST_TEXT="${MIXED_REQUEST_TEXT:-知名爆料人 Moores Law is Dead 在近期的视频中表示，PlayStation 5 Pro 不带光驱的型号价格有望低至 500 美元。长时间。肥差}"
MIXED_REQUEST_VOICE="${MIXED_REQUEST_VOICE:-ZH_MIX_EN}"
MIXED_REQUEST_FORMAT="${MIXED_REQUEST_FORMAT:-mp3}"
MIXED_REQUEST_SPEED="${MIXED_REQUEST_SPEED:-1.0}"
VOICE_MATRIX_TEST="${VOICE_MATRIX_TEST:-1}"
VOICE_MATRIX_VOICES="${VOICE_MATRIX_VOICES:-alloy,echo,fable,onyx,nova,shimmer,ash,ballad,coral,sage,verse,ZH_MIX_EN,ZH,EN,JP,KR,ES,FR}"
VOICE_MATRIX_FORMAT="${VOICE_MATRIX_FORMAT:-mp3}"
VOICE_MATRIX_SPEED="${VOICE_MATRIX_SPEED:-1.0}"
VOICE_MATRIX_STRICT="${VOICE_MATRIX_STRICT:-1}"
VOICE_MATRIX_ASR_VERIFY="${VOICE_MATRIX_ASR_VERIFY:-auto}"
CHECK_OFFLINE_LOG_ERRORS="${CHECK_OFFLINE_LOG_ERRORS:-1}"
EXTERNAL_SPEECH_URL="${EXTERNAL_SPEECH_URL:-}"
EXTERNAL_REQUEST_TEXT="${EXTERNAL_REQUEST_TEXT:-长时间。肥差}"
EXTERNAL_REQUEST_VOICE="${EXTERNAL_REQUEST_VOICE:-alloy}"
EXTERNAL_REQUEST_FORMAT="${EXTERNAL_REQUEST_FORMAT:-mp3}"
EXTERNAL_REQUEST_SPEED="${EXTERNAL_REQUEST_SPEED:-1.0}"
ASR_VERIFY_URL="${ASR_VERIFY_URL:-}"
ASR_MODEL="${ASR_MODEL:-whisper-1}"
ASR_RESPONSE_FORMAT="${ASR_RESPONSE_FORMAT:-json}"
ASR_API_KEY="${ASR_API_KEY:-}"
ASR_EXPECT_TEXT="${ASR_EXPECT_TEXT:-}"
ASR_MIN_RATIO="${ASR_MIN_RATIO:-0.60}"
ASR_STRICT="${ASR_STRICT:-1}"
ASR_AUDIO_TARGET="${ASR_AUDIO_TARGET:-auto}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
MAX_HEALTH_RETRIES="${MAX_HEALTH_RETRIES:-60}"
HEALTH_RETRY_INTERVAL="${HEALTH_RETRY_INTERVAL:-2}"
KEEP_CONTAINER="${KEEP_CONTAINER:-0}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/jetson-e2e-${TS}}"
mkdir -p "${LOG_DIR}"

SSH_ARGS=("-o" "BatchMode=yes" "-o" "ConnectTimeout=10")
if [[ -n "${SSH_EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_opts=(${SSH_EXTRA_OPTS})
  SSH_ARGS+=("${extra_opts[@]}")
fi

RSYNC_ARGS=(
  "-rtP"
  "--exclude" ".venv"
  "--exclude" ".git"
  "--exclude" "logs"
  "--exclude" "__pycache__"
  "--exclude" "*/__pycache__"
)
if [[ -n "${RSYNC_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_rsync=(${RSYNC_EXTRA_ARGS})
  RSYNC_ARGS+=("${extra_rsync[@]}")
fi

json_escape() {
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  s=${s//$'\t'/\\t}
  printf '%s' "$s"
}

sanitize_token() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_'
}

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_falsy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

compare_asr_response() {
  local resp_file="$1"
  local expect_text="$2"
  local min_ratio="$3"
  python3 -c 'import json, sys, re, difflib, pathlib
p = pathlib.Path(sys.argv[1])
raw = p.read_text(encoding="utf-8", errors="ignore").strip()
txt = raw
if raw.startswith("{"):
    try:
        txt = str(json.loads(raw).get("text", "")).strip()
    except Exception:
        txt = raw
clean_txt = re.sub(r"<\|.*?\|>", "", txt).strip()
expect = sys.argv[2]
min_ratio = float(sys.argv[3])
def norm(s):
    return re.sub(r"[\s\W_]+", "", s.lower(), flags=re.UNICODE)
n_txt = norm(clean_txt)
n_expect = norm(expect)
ratio = difflib.SequenceMatcher(None, n_txt, n_expect).ratio() if (n_txt or n_expect) else 1.0
print(f"asr_text={txt}")
print(f"asr_text_clean={clean_txt}")
print(f"expect_text={expect}")
print(f"asr_ratio={ratio:.4f}")
print(f"asr_min_ratio={min_ratio:.4f}")
sys.exit(0 if ratio >= min_ratio else 2)' "$resp_file" "$expect_text" "$min_ratio"
}

run_asr_verify() {
  local log_name="$1"
  local audio_path="$2"
  local expect_text="$3"
  local resp_file="$4"
  local verify_file="$5"

  if [[ ! -f "${audio_path}" ]]; then
    log "[error] ASR source audio not found: ${audio_path}"
    return 1
  fi

  local asr_cmd=(curl -fsS -X POST "${ASR_VERIFY_URL}" -F "file=@${audio_path}" -F "model=${ASR_MODEL}" -F "response_format=${ASR_RESPONSE_FORMAT}" --output "${resp_file}")
  if [[ -n "${ASR_API_KEY}" ]]; then
    asr_cmd+=(-H "Authorization: Bearer ${ASR_API_KEY}")
  fi
  run_local "${log_name}" "${asr_cmd[@]}"

  compare_asr_response "${resp_file}" "${expect_text}" "${ASR_MIN_RATIO}" | tee "${verify_file}"
  return $?
}

voice_sample_text() {
  local voice_upper
  voice_upper="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  case "${voice_upper}" in
    ZH|ZH_MIX_EN)
      printf '%s' "长时间。肥差"
      ;;
    EN|ALLOY|ECHO|FABLE|ONYX|NOVA|SHIMMER|ASH|BALLAD|CORAL|SAGE|VERSE)
      printf '%s' "hello world"
      ;;
    JP)
      printf '%s' "こんにちは"
      ;;
    KR)
      printf '%s' "안녕하세요"
      ;;
    ES)
      printf '%s' "hola"
      ;;
    FR)
      printf '%s' "bonjour"
      ;;
    *)
      printf '%s' "MeloTTS voice matrix test."
      ;;
  esac
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_local() {
  local name="$1"
  shift
  log "[local:${name}] $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_remote() {
  local name="$1"
  local script="$2"
  log "[remote:${name}] ${script}"
  ssh "${SSH_ARGS[@]}" "${REMOTE}" "bash -lc $(printf '%q' "$script")" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

capture_remote() {
  local name="$1"
  local script="$2"
  if ! ssh "${SSH_ARGS[@]}" "${REMOTE}" "bash -lc $(printf '%q' "$script")" >"${LOG_DIR}/${name}.log" 2>&1; then
    log "[warn] capture ${name} failed"
  fi
}


write_test_md() {
  local status="$1"
  local test_md="${REPO_ROOT}/TEST.md"
  local voice_summary="${LOG_DIR}/voice_matrix/summary.txt"
  local main_asr_verify="${LOG_DIR}/asr_verify.txt"
  local first_matrix_asr_verify=""
  local health_status="SKIP"
  local basic_status="FAIL"
  local mixed_status="SKIP"
  local voice_matrix_status="SKIP"
  local voice_matrix_failures="n/a"
  local voice_matrix_asr_failures="n/a"
  local main_asr_status="SKIP"
  local failure_hint=""
  local note_line=""

  if [[ -d "${LOG_DIR}/voice_matrix" ]]; then
    first_matrix_asr_verify="$(find "${LOG_DIR}/voice_matrix" -maxdepth 1 -type f -name 'asr_verify_*.txt' | sort -V | head -n1)"
  fi

  if [[ -f "${LOG_DIR}/wait_health.log" ]] && grep -q 'health_ok_try=' "${LOG_DIR}/wait_health.log"; then
    health_status="PASS"
  elif [[ -f "${LOG_DIR}/wait_health.log" ]]; then
    health_status="FAIL"
  fi

  if [[ -s "${LOG_DIR}/response.${REQUEST_FORMAT}" ]]; then
    basic_status="PASS"
  fi

  if [[ "${MIXED_LANG_TEST}" == "1" ]]; then
    mixed_status="FAIL"
    if [[ -s "${LOG_DIR}/mixed_response.${MIXED_REQUEST_FORMAT}" ]]; then
      mixed_status="PASS"
    fi
  fi

  if [[ -f "${voice_summary}" ]]; then
    voice_matrix_status="INCOMPLETE"
    if grep -q '^result=' "${voice_summary}"; then
      voice_matrix_status="$(awk -F= '/^result=/{print $2; exit}' "${voice_summary}")"
    elif [[ "${status}" != "PASS" ]]; then
      voice_matrix_status="FAIL"
    fi
    voice_matrix_failures="$(awk -F= '/^total_failures=/{print $2; exit}' "${voice_summary}" 2>/dev/null || true)"
    voice_matrix_asr_failures="$(awk -F= '/^asr_failures=/{print $2; exit}' "${voice_summary}" 2>/dev/null || true)"
    [[ -n "${voice_matrix_failures}" ]] || voice_matrix_failures="n/a"
    [[ -n "${voice_matrix_asr_failures}" ]] || voice_matrix_asr_failures="n/a"
  fi

  if [[ -n "${ASR_VERIFY_URL}" ]]; then
    if [[ -f "${main_asr_verify}" ]]; then
      if grep -q '^asr_ratio=' "${main_asr_verify}"; then
        main_asr_status="PASS"
      else
        main_asr_status="FAIL"
      fi
    elif [[ -n "${first_matrix_asr_verify}" ]]; then
      main_asr_status="MATRIX_ONLY"
    else
      main_asr_status="FAIL"
    fi
  fi

  if [[ -f "${LOG_DIR}/summary.txt" ]]; then
    failure_hint="$(sed -n 's/^error_hint=//p' "${LOG_DIR}/summary.txt" | head -n1)"
  fi

  if [[ -n "${first_matrix_asr_verify}" ]]; then
    note_line="$(paste -sd ';' "${first_matrix_asr_verify}" | sed 's/;/; /g')"
  fi

  {
    printf '%s
' '# TEST'
    printf '%s
' ''
    printf '%s
' "- Updated: $(date '+%F %T %Z')"
    printf '%s
' "- Status: ${status}"
    printf '%s
' '- Script: ./scripts/jetson_e2e_test.sh'
    printf '%s
' "- Remote: ${REMOTE}"
    printf '%s
' "- Image: ${IMAGE}"
    printf '%s
' "- Container: ${CONTAINER_NAME}"
    printf '%s
' "- Log Dir: ${LOG_DIR}"
    printf '%s
' "- Offline Mode: ${OFFLINE_MODE}"
    printf '%s
' "- ASR Verify URL: ${ASR_VERIFY_URL:-unset}"
    printf '%s
' ''
    printf '%s
' '## Checks'
    printf '%s
' ''
    printf '%s
' "- Health: ${health_status}"
    printf '%s
' "- Basic Speech API: ${basic_status}"
    printf '%s
' "- Mixed ZH_MIX_EN: ${mixed_status}"
    printf '%s
' "- Voice Matrix: ${voice_matrix_status}"
    printf '%s
' "- Voice Matrix Total Failures: ${voice_matrix_failures}"
    printf '%s
' "- Voice Matrix ASR Failures: ${voice_matrix_asr_failures}"
    printf '%s
' "- Main ASR Check: ${main_asr_status}"
    printf '%s
' ''
    printf '%s
' '## Notes'
    if [[ -n "${note_line}" ]]; then
      printf '%s
' "- First matrix ASR result: ${note_line}"
    fi
    if [[ -n "${failure_hint}" ]]; then
      printf '%s
' "- Failure hint: ${failure_hint}"
    fi
    printf '%s
' "- Summary file: ${LOG_DIR}/summary.txt"
    if [[ -f "${voice_summary}" ]]; then
      printf '%s
' "- Voice matrix summary: ${voice_summary}"
    fi
    if [[ -f "${main_asr_verify}" ]]; then
      printf '%s
' "- Main ASR verify: ${main_asr_verify}"
    fi
  } >"${test_md}"
}

cleanup() {
  local rc=$?
  capture_remote "docker_ps" "docker ps -a --filter name=^/${CONTAINER_NAME}$"
  capture_remote "docker_logs" "docker logs ${CONTAINER_NAME} 2>&1 || true"
  capture_remote "final_health" "curl -sS http://127.0.0.1:${SERVICE_PORT}${HEALTH_PATH} || true"

  if [[ "${KEEP_CONTAINER}" != "1" ]]; then
    capture_remote "docker_cleanup" "docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true"
  fi

  if [[ $rc -eq 0 ]]; then
    printf 'PASS\nremote=%s\nimage=%s\ncontainer=%s\nlogs=%s\n' \
      "${REMOTE}" "${IMAGE}" "${CONTAINER_NAME}" "${LOG_DIR}" | tee "${LOG_DIR}/summary.txt"
    write_test_md "PASS"
    log "Jetson e2e test succeeded. Logs: ${LOG_DIR}"
  else
    {
      printf 'FAIL\n'
      printf 'remote=%s\n' "${REMOTE}"
      printf 'image=%s\n' "${IMAGE}"
      printf 'container=%s\n' "${CONTAINER_NAME}"
      printf 'logs=%s\n' "${LOG_DIR}"
      if [[ -s "${LOG_DIR}/docker_logs.log" ]]; then
        printf 'error_hint=%s\n' "$(tail -n 1 "${LOG_DIR}/docker_logs.log" | tr -d '\r')"
      fi
    } | tee "${LOG_DIR}/summary.txt"

    write_test_md "FAIL"
    log "Jetson e2e test failed (exit=${rc}). Logs: ${LOG_DIR}"
    if [[ -s "${LOG_DIR}/docker_logs.log" ]]; then
      log "Last container log lines:"
      tail -n 20 "${LOG_DIR}/docker_logs.log" || true
    fi
  fi
  exit $rc
}
trap cleanup EXIT

REQUEST_JSON="${LOG_DIR}/request.json"
printf '{"model":"tts-1","input":"%s","voice":"%s","response_format":"%s","speed":%s}\n' \
  "$(json_escape "${REQUEST_TEXT}")" \
  "$(json_escape "${REQUEST_VOICE}")" \
  "$(json_escape "${REQUEST_FORMAT}")" \
  "${REQUEST_SPEED}" >"${REQUEST_JSON}"

EXTERNAL_REQUEST_JSON="${LOG_DIR}/external_request.json"
printf '{"model":"tts-1","input":"%s","voice":"%s","response_format":"%s","speed":%s}\n' \
  "$(json_escape "${EXTERNAL_REQUEST_TEXT}")" \
  "$(json_escape "${EXTERNAL_REQUEST_VOICE}")" \
  "$(json_escape "${EXTERNAL_REQUEST_FORMAT}")" \
  "${EXTERNAL_REQUEST_SPEED}" >"${EXTERNAL_REQUEST_JSON}"

MIXED_REQUEST_JSON="${LOG_DIR}/mixed_request.json"
printf '{"model":"tts-1","input":"%s","voice":"%s","response_format":"%s","speed":%s}\n' \
  "$(json_escape "${MIXED_REQUEST_TEXT}")" \
  "$(json_escape "${MIXED_REQUEST_VOICE}")" \
  "$(json_escape "${MIXED_REQUEST_FORMAT}")" \
  "${MIXED_REQUEST_SPEED}" >"${MIXED_REQUEST_JSON}"

log "Step 1/6: rsync code to ${REMOTE}:${REMOTE_DIR}"
if [[ "${SKIP_SYNC}" == "1" ]]; then
  log "[skip] SKIP_SYNC=1"
else
  run_remote "prepare_remote_dir" "mkdir -p $(printf '%q' "${REMOTE_DIR}")"
  run_local "rsync" rsync "${RSYNC_ARGS[@]}" ./ "${REMOTE}:${REMOTE_DIR}/"
fi

log "Step 2/6: build docker image on Jetson via make build_jetson"
if [[ "${SKIP_BUILD}" == "1" ]]; then
  log "[skip] SKIP_BUILD=1"
else
  run_remote "build_jetson" "cd $(printf '%q' "${REMOTE_DIR}") && make build_jetson"
fi

log "Step 3/6: run docker container on Jetson"
DOCKER_RUN_EFFECTIVE_ARGS="${DOCKER_RUN_ARGS}"
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  DOCKER_RUN_EFFECTIVE_ARGS="${DOCKER_RUN_EFFECTIVE_ARGS} ${OFFLINE_ENV_ARGS}"
  log "[info] OFFLINE_MODE=1 (default), enforce offline runtime envs"
else
  log "[warn] OFFLINE_MODE=0, container may access network resources"
fi
run_remote "docker_run" "docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true; docker run -d --name ${CONTAINER_NAME} ${DOCKER_RUN_EFFECTIVE_ARGS} ${IMAGE}"
run_remote "docker_status" "docker ps --filter name=^/${CONTAINER_NAME}$ --format '{{.ID}} {{.Image}} {{.Status}}'"
run_remote "docker_status_all" "docker ps -a --filter name=^/${CONTAINER_NAME}$ --format '{{.ID}} {{.Image}} {{.Status}}'"

VOICE_MATRIX_ASR_ENABLED=0
if [[ -n "${ASR_VERIFY_URL}" ]]; then
  if is_truthy "${VOICE_MATRIX_ASR_VERIFY}" || [[ "${VOICE_MATRIX_ASR_VERIFY}" == "auto" ]]; then
    VOICE_MATRIX_ASR_ENABLED=1
  elif ! is_falsy "${VOICE_MATRIX_ASR_VERIFY}"; then
    log "[error] invalid VOICE_MATRIX_ASR_VERIFY=${VOICE_MATRIX_ASR_VERIFY} (expected auto/0/1)"
    exit 1
  fi
elif is_truthy "${VOICE_MATRIX_ASR_VERIFY}"; then
  log "[error] VOICE_MATRIX_ASR_VERIFY=1 requires ASR_VERIFY_URL"
  exit 1
elif [[ "${VOICE_MATRIX_ASR_VERIFY}" != "auto" ]] && ! is_falsy "${VOICE_MATRIX_ASR_VERIFY}"; then
  log "[error] invalid VOICE_MATRIX_ASR_VERIFY=${VOICE_MATRIX_ASR_VERIFY} (expected auto/0/1)"
  exit 1
fi

log "Step 4/6: wait for HTTP health check ${HEALTH_PATH}"
run_remote "wait_health" "for i in \$(seq 1 ${MAX_HEALTH_RETRIES}); do if curl -fsS http://127.0.0.1:${SERVICE_PORT}${HEALTH_PATH} >/dev/null; then echo health_ok_try=\$i; exit 0; fi; if ! docker ps --filter name=^/${CONTAINER_NAME}$ --filter status=running --format '{{.ID}}' | grep -q .; then echo container_not_running >&2; docker logs ${CONTAINER_NAME} >&2 || true; exit 1; fi; sleep ${HEALTH_RETRY_INTERVAL}; done; echo health_check_failed >&2; docker logs ${CONTAINER_NAME} >&2 || true; exit 1"

log "Step 5/6: call speech HTTP API ${SPEECH_PATH}"
run_local "push_request_json" scp "${SSH_ARGS[@]}" "${REQUEST_JSON}" "${REMOTE}:/tmp/${CONTAINER_NAME}.request.json"
run_remote "speech_request" "curl -fsS -X POST http://127.0.0.1:${SERVICE_PORT}${SPEECH_PATH} -H 'Content-Type: application/json' --data @/tmp/${CONTAINER_NAME}.request.json -D /tmp/${CONTAINER_NAME}.headers -o /tmp/${CONTAINER_NAME}.wav && test -s /tmp/${CONTAINER_NAME}.wav && ls -lh /tmp/${CONTAINER_NAME}.wav"

if [[ "${MIXED_LANG_TEST}" == "1" ]]; then
  log "Step 5.1/6: mixed-language offline request (${MIXED_REQUEST_VOICE})"
  run_local "push_mixed_request_json" scp "${SSH_ARGS[@]}" "${MIXED_REQUEST_JSON}" "${REMOTE}:/tmp/${CONTAINER_NAME}.mixed.request.json"
  run_remote "speech_request_mixed" "curl -fsS -X POST http://127.0.0.1:${SERVICE_PORT}${SPEECH_PATH} -H 'Content-Type: application/json' --data @/tmp/${CONTAINER_NAME}.mixed.request.json -D /tmp/${CONTAINER_NAME}.mixed.headers -o /tmp/${CONTAINER_NAME}.mixed.${MIXED_REQUEST_FORMAT} && test -s /tmp/${CONTAINER_NAME}.mixed.${MIXED_REQUEST_FORMAT} && ls -lh /tmp/${CONTAINER_NAME}.mixed.${MIXED_REQUEST_FORMAT}"
fi

if [[ "${VOICE_MATRIX_TEST}" == "1" ]]; then
  log "Step 5.2/6: voice matrix generation test"
  VOICE_MATRIX_DIR="${LOG_DIR}/voice_matrix"
  mkdir -p "${VOICE_MATRIX_DIR}"
  VOICE_MATRIX_SUMMARY="${VOICE_MATRIX_DIR}/summary.txt"
  : > "${VOICE_MATRIX_SUMMARY}"
  printf 'voice_matrix_asr=%s\nasr_verify_url=%s\n\n' \
    "$([[ ${VOICE_MATRIX_ASR_ENABLED} -eq 1 ]] && printf 'enabled' || printf 'disabled')" \
    "${ASR_VERIFY_URL:-unset}" >>"${VOICE_MATRIX_SUMMARY}"
  matrix_fail_count=0
  matrix_gen_fail_count=0
  matrix_asr_fail_count=0
  idx=0
  IFS=',' read -r -a voice_items <<<"${VOICE_MATRIX_VOICES}"
  for raw_voice in "${voice_items[@]}"; do
    voice="$(printf '%s' "${raw_voice}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${voice}" ]] || continue
    idx=$((idx + 1))
    voice_safe="$(sanitize_token "${voice}")"
    sample_text="$(voice_sample_text "${voice}")"
    req_local="${VOICE_MATRIX_DIR}/request_${idx}_${voice_safe}.json"
    out_local="${VOICE_MATRIX_DIR}/response_${idx}_${voice_safe}.${VOICE_MATRIX_FORMAT}"
    hdr_local="${VOICE_MATRIX_DIR}/response_${idx}_${voice_safe}.headers"
    req_remote="/tmp/${CONTAINER_NAME}.voice.${idx}.${voice_safe}.request.json"
    out_remote="/tmp/${CONTAINER_NAME}.voice.${idx}.${voice_safe}.${VOICE_MATRIX_FORMAT}"
    hdr_remote="/tmp/${CONTAINER_NAME}.voice.${idx}.${voice_safe}.headers"

    printf '{"model":"tts-1","input":"%s","voice":"%s","response_format":"%s","speed":%s}\n' \
      "$(json_escape "${sample_text}")" \
      "$(json_escape "${voice}")" \
      "$(json_escape "${VOICE_MATRIX_FORMAT}")" \
      "${VOICE_MATRIX_SPEED}" >"${req_local}"

    run_local "push_voice_request_${idx}_${voice_safe}" scp "${SSH_ARGS[@]}" "${req_local}" "${REMOTE}:${req_remote}"
    set +e
    run_remote "speech_request_voice_${idx}_${voice_safe}" "curl -fsS -X POST http://127.0.0.1:${SERVICE_PORT}${SPEECH_PATH} -H 'Content-Type: application/json' --data @${req_remote} -D ${hdr_remote} -o ${out_remote} && test -s ${out_remote} && ls -lh ${out_remote}"
    voice_rc=$?
    set -e

    if [[ ${voice_rc} -eq 0 ]]; then
      run_local "pull_voice_audio_${idx}_${voice_safe}" scp "${SSH_ARGS[@]}" "${REMOTE}:${out_remote}" "${out_local}"
      run_local "pull_voice_headers_${idx}_${voice_safe}" scp "${SSH_ARGS[@]}" "${REMOTE}:${hdr_remote}" "${hdr_local}"
      if [[ ${VOICE_MATRIX_ASR_ENABLED} -eq 1 ]]; then
        asr_resp_local="${VOICE_MATRIX_DIR}/asr_response_${idx}_${voice_safe}.${ASR_RESPONSE_FORMAT}"
        asr_verify_local="${VOICE_MATRIX_DIR}/asr_verify_${idx}_${voice_safe}.txt"
        set +e
        run_asr_verify "voice_asr_${idx}_${voice_safe}" "${out_local}" "${sample_text}" "${asr_resp_local}" "${asr_verify_local}"
        voice_asr_rc=$?
        set -e

        if [[ ${voice_asr_rc} -eq 0 ]]; then
          voice_asr_ratio="$(awk -F= '/^asr_ratio=/{print $2; exit}' "${asr_verify_local}")"
          echo "PASS voice=${voice} file=${out_local} asr=PASS ratio=${voice_asr_ratio:-unknown} verify=${asr_verify_local}" >>"${VOICE_MATRIX_SUMMARY}"
        else
          matrix_fail_count=$((matrix_fail_count + 1))
          matrix_asr_fail_count=$((matrix_asr_fail_count + 1))
          echo "FAIL voice=${voice} file=${out_local} asr=FAIL rc=${voice_asr_rc} verify=${asr_verify_local}" >>"${VOICE_MATRIX_SUMMARY}"
          log "[warn] voice matrix ASR verify failed for voice=${voice} (rc=${voice_asr_rc})"
        fi
      else
        echo "PASS voice=${voice} file=${out_local} asr=SKIP" >>"${VOICE_MATRIX_SUMMARY}"
      fi
    else
      matrix_fail_count=$((matrix_fail_count + 1))
      matrix_gen_fail_count=$((matrix_gen_fail_count + 1))
      echo "FAIL voice=${voice} rc=${voice_rc}" >>"${VOICE_MATRIX_SUMMARY}"
      capture_remote "voice_matrix_fail_${idx}_${voice_safe}_docker_logs" "docker logs ${CONTAINER_NAME} | tail -n 120"
      log "[warn] voice matrix failed for voice=${voice} (rc=${voice_rc})"
    fi
    capture_remote "cleanup_voice_tmp_${idx}_${voice_safe}" "rm -f ${req_remote} ${out_remote} ${hdr_remote}"
  done

  printf '\nresult=%s\ntotal_failures=%s\ngeneration_failures=%s\nasr_failures=%s\n' \
    "$([[ ${matrix_fail_count} -eq 0 ]] && printf 'PASS' || printf 'FAIL')" \
    "${matrix_fail_count}" \
    "${matrix_gen_fail_count}" \
    "${matrix_asr_fail_count}" >>"${VOICE_MATRIX_SUMMARY}"

  if [[ ${matrix_fail_count} -gt 0 ]]; then
    log "[warn] voice matrix failures=${matrix_fail_count} (gen=${matrix_gen_fail_count}, asr=${matrix_asr_fail_count}), summary=${VOICE_MATRIX_SUMMARY}"
    if [[ "${VOICE_MATRIX_STRICT}" == "1" ]]; then
      exit 21
    fi
  else
    log "[info] voice matrix all passed, summary=${VOICE_MATRIX_SUMMARY}"
  fi
fi

if [[ -n "${EXTERNAL_SPEECH_URL}" ]]; then
  log "Step 5.3/6: call external speech HTTP API ${EXTERNAL_SPEECH_URL}"
  run_local "external_speech_request" curl -fsS -X POST "${EXTERNAL_SPEECH_URL}" \
    -H "Content-Type: application/json" \
    --data @"${EXTERNAL_REQUEST_JSON}" \
    --output "${LOG_DIR}/external_speech.${EXTERNAL_REQUEST_FORMAT}"
  run_local "external_speech_size" ls -lh "${LOG_DIR}/external_speech.${EXTERNAL_REQUEST_FORMAT}"
fi

log "Step 6/6: collect test artifacts and feedback"
run_local "pull_response_wav" scp "${SSH_ARGS[@]}" "${REMOTE}:/tmp/${CONTAINER_NAME}.wav" "${LOG_DIR}/response.${REQUEST_FORMAT}"
run_local "pull_response_headers" scp "${SSH_ARGS[@]}" "${REMOTE}:/tmp/${CONTAINER_NAME}.headers" "${LOG_DIR}/response.headers"
if [[ "${MIXED_LANG_TEST}" == "1" ]]; then
  run_local "pull_mixed_response" scp "${SSH_ARGS[@]}" "${REMOTE}:/tmp/${CONTAINER_NAME}.mixed.${MIXED_REQUEST_FORMAT}" "${LOG_DIR}/mixed_response.${MIXED_REQUEST_FORMAT}"
  run_local "pull_mixed_headers" scp "${SSH_ARGS[@]}" "${REMOTE}:/tmp/${CONTAINER_NAME}.mixed.headers" "${LOG_DIR}/mixed_response.headers"
fi
run_remote "remote_cleanup_tmp" "rm -f /tmp/${CONTAINER_NAME}.request.json /tmp/${CONTAINER_NAME}.headers /tmp/${CONTAINER_NAME}.wav /tmp/${CONTAINER_NAME}.mixed.request.json /tmp/${CONTAINER_NAME}.mixed.headers /tmp/${CONTAINER_NAME}.mixed.${MIXED_REQUEST_FORMAT}"

if [[ "${CHECK_OFFLINE_LOG_ERRORS}" == "1" ]]; then
  log "Step 6.05/6: scan runtime logs for offline download errors"
  set +e
  run_remote "offline_log_scan" "docker logs ${CONTAINER_NAME} 2>&1 | grep -E 'LocalEntryNotFoundError|couldn.t connect to .https://huggingface.co.|Offline mode enabled but tokenizer' && exit 2 || exit 0"
  offline_scan_rc=$?
  set -e
  if [[ ${offline_scan_rc} -ne 0 ]]; then
    log "[error] runtime log contains offline download/cache errors"
    exit ${offline_scan_rc}
  fi
fi

if [[ -n "${ASR_VERIFY_URL}" ]]; then
  log "Step 6.1/6: ASR transcription verify via ${ASR_VERIFY_URL}"

  asr_audio_path="${LOG_DIR}/response.${REQUEST_FORMAT}"
  asr_expect_text="${REQUEST_TEXT}"
  if [[ "${ASR_AUDIO_TARGET}" == "external" ]]; then
    asr_audio_path="${LOG_DIR}/external_speech.${EXTERNAL_REQUEST_FORMAT}"
    asr_expect_text="${EXTERNAL_REQUEST_TEXT}"
  elif [[ "${ASR_AUDIO_TARGET}" == "auto" && -f "${LOG_DIR}/external_speech.${EXTERNAL_REQUEST_FORMAT}" ]]; then
    asr_audio_path="${LOG_DIR}/external_speech.${EXTERNAL_REQUEST_FORMAT}"
    asr_expect_text="${EXTERNAL_REQUEST_TEXT}"
  fi
  if [[ -n "${ASR_EXPECT_TEXT}" ]]; then
    asr_expect_text="${ASR_EXPECT_TEXT}"
  fi
  if [[ ! -f "${asr_audio_path}" ]]; then
    log "[error] ASR source audio not found: ${asr_audio_path}"
    exit 1
  fi

  ASR_RESP_FILE="${LOG_DIR}/asr_response.${ASR_RESPONSE_FORMAT}"
  set +e
  run_asr_verify "asr_transcribe" "${asr_audio_path}" "${asr_expect_text}" "${ASR_RESP_FILE}" "${LOG_DIR}/asr_verify.txt"
  asr_rc=$?
  set -e

  if [[ ${asr_rc} -ne 0 ]]; then
    if [[ "${ASR_STRICT}" == "1" ]]; then
      log "[error] ASR verify failed (rc=${asr_rc})"
      exit ${asr_rc}
    fi
    log "[warn] ASR verify failed (rc=${asr_rc}), but ASR_STRICT=0 so continue"
  fi
fi
