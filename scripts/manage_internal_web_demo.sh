#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RS_AGENT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUNTIME_DIR="${RS_AGENT_WEB_RUNTIME_DIR:-${TMPDIR:-/tmp}/rs-agent-web}"
WEB_BIN="${RS_AGENT_WEB_BIN:-rs-agent-web}"
ADDRESS="${RS_AGENT_WEB_ADDRESS:-127.0.0.1}"
PORT="${RS_AGENT_WEB_PORT:-8501}"
MANIFEST="${RS_AGENT_DEMO_MANIFEST:-}"
WORK_DIR="${RS_AGENT_WEB_WORKDIR:-${RUNTIME_DIR}/work}"
ARTIFACT_DIR="${RS_AGENT_WEB_ARTIFACT_DIR:-${RUNTIME_DIR}/artifacts}"
PID_FILE="${RUNTIME_DIR}/streamlit.pid"
LOG_FILE="${RUNTIME_DIR}/streamlit.log"

if [[ "${ADDRESS}" != "127.0.0.1" && "${ADDRESS}" != "localhost" && "${ADDRESS}" != "::1" ]]; then
  echo "Refusing non-loopback address: ${ADDRESS}" >&2
  exit 2
fi

is_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(<"${PID_FILE}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

start() {
  mkdir -p "${RUNTIME_DIR}" "${WORK_DIR}" "${ARTIFACT_DIR}"
  if is_running; then
    echo "RS-Agent Web Demo is already running (pid $(<"${PID_FILE}"))."
    return
  fi
  if [[ -z "${MANIFEST}" || ! -f "${MANIFEST}" ]]; then
    echo "Set RS_AGENT_DEMO_MANIFEST to an existing JSONL manifest." >&2
    exit 2
  fi

  nohup env \
    RS_AGENT_REPO_ROOT="${REPO_ROOT}" \
    RS_AGENT_DEMO_MANIFEST="${MANIFEST}" \
    RS_AGENT_WEB_WORKDIR="${WORK_DIR}" \
    RS_AGENT_WEB_ARTIFACT_DIR="${ARTIFACT_DIR}" \
    RS_AGENT_WEB_PASSWORD="${RS_AGENT_WEB_PASSWORD:-}" \
    "${WEB_BIN}" \
      --server.address "${ADDRESS}" \
      --server.port "${PORT}" \
      --server.headless true \
      >"${LOG_FILE}" 2>&1 </dev/null &
  echo "$!" >"${PID_FILE}"

  for _ in {1..20}; do
    if ! is_running; then
      echo "RS-Agent Web Demo exited during startup." >&2
      tail -40 "${LOG_FILE}" >&2 || true
      exit 1
    fi
    if curl -fsS "http://${ADDRESS}:${PORT}/_stcore/health" >/dev/null 2>&1; then
      echo "RS-Agent Web Demo is healthy at http://${ADDRESS}:${PORT} (pid $(<"${PID_FILE}"))."
      return
    fi
    sleep 0.5
  done

  echo "RS-Agent Web Demo did not become healthy in time." >&2
  tail -40 "${LOG_FILE}" >&2 || true
  exit 1
}

status() {
  if ! is_running; then
    echo "RS-Agent Web Demo is stopped."
    return 1
  fi
  local health="unhealthy"
  if curl -fsS "http://${ADDRESS}:${PORT}/_stcore/health" >/dev/null 2>&1; then
    health="healthy"
  fi
  echo "RS-Agent Web Demo is ${health} at http://${ADDRESS}:${PORT} (pid $(<"${PID_FILE}"))."
}

stop() {
  if ! is_running; then
    echo "RS-Agent Web Demo is already stopped."
    return
  fi
  local pid
  pid="$(<"${PID_FILE}")"
  kill "${pid}"
  for _ in {1..20}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "RS-Agent Web Demo stopped."
      return
    fi
    sleep 0.5
  done
  echo "RS-Agent Web Demo did not stop within 10 seconds (pid ${pid})." >&2
  exit 1
}

case "${ACTION}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *)
    echo "Usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
