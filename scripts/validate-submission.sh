#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/4: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/4: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/4: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/4: Verifying grader scores via smoke-test episode${NC} ..."

# Run a minimal invoice_classifier episode via the stateful /ui endpoints and
# confirm the terminal obs.score is strictly in (0, 1).
# python3 is available wherever openenv-core is installed.
SMOKE_PY=$(portable_mktemp "smoke")
CLEANUP_FILES+=("$SMOKE_PY")
cat > "$SMOKE_PY" << 'PYEOF'
import sys, json
try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError
except ImportError:
    print("SKIP:no_urllib", flush=True)
    sys.exit(0)

base = sys.argv[1].rstrip("/")

def post(path, payload):
    body = json.dumps(payload).encode()
    req = Request(f"{base}{path}", data=body,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except URLError as exc:
        print(f"ERROR:{exc}", flush=True)
        sys.exit(1)

# Reset task 1 via the stateful UI endpoint
obs = post("/ui/reset", {"task": "invoice_classifier"})
pending = obs.get("pending_actions", [])
if not pending:
    print("ERROR:no_pending_actions", flush=True)
    sys.exit(1)

# Classify all invoices (wrong type is fine — we only care that the episode ends)
final_obs = None
for inv_id in pending:
    o = post("/ui/step", {
        "action_type": "classify_invoice",
        "invoice_id": inv_id,
        "invoice_type": "B2B",
        "hsn_code": "8471",
    })
    if o.get("done"):
        final_obs = o
        break

if final_obs is None:
    print("ERROR:episode_never_ended", flush=True)
    sys.exit(1)

score = final_obs.get("score")
if score is None:
    print("NULL:score_field_missing_or_null", flush=True)
    sys.exit(1)

try:
    s = float(score)
except (TypeError, ValueError):
    print(f"INVALID:{score}", flush=True)
    sys.exit(1)

if 0.0 < s < 1.0:
    print(f"OK:{s}", flush=True)
else:
    print(f"OUT_OF_RANGE:{s}", flush=True)
    sys.exit(1)
PYEOF

SMOKE_OK=false
SMOKE_OUTPUT=$(python3 "$SMOKE_PY" "$PING_URL" 2>/dev/null) && SMOKE_OK=true

if [ "$SMOKE_OK" = true ]; then
  case "$SMOKE_OUTPUT" in
    OK:*)
      SCORE_VAL="${SMOKE_OUTPUT#OK:}"
      pass "Grader score $SCORE_VAL is strictly within (0, 1)"
      ;;
    SKIP:*)
      pass "Grader score check skipped (${SMOKE_OUTPUT#SKIP:})"
      ;;
    *)
      fail "Grader score check returned unexpected output: $SMOKE_OUTPUT"
      stop_at "Step 4"
      ;;
  esac
else
  fail "Grader score check failed: $SMOKE_OUTPUT"
  hint "obs.score must be a float strictly in (0, 1) on the terminal observation."
  hint "Check that graders return _clamp(score) and obs.score is set when done=True."
  stop_at "Step 4"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
