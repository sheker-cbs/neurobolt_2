#!/bin/bash
# Submit Algermissen subjects the SHiVAi way: one sbatch per batch file,
# no --array, no nested sbatch. Each job runs subjects sequentially on one GPU.
#
# Flow:
#   1) this script writes subject_batches/batch_NNN.txt
#   2) sbatch --job-name=… run_neuro.sh <batchfile>
#      run_neuro.sh exports NEUROBOLT_SUBJECTS; main.py trains that list.
#
# Default (new submit): creates one dated campaign folder for the whole wave:
#   {CHECKPOINT_PATH}/log_baseline/YYYYMMDD-HHMMSS/   (or log_attn/)
#   ├── campaign_meta.json
#   └── runs/{ts}_j{JOB}_{mode}_{subject}/…
# All jobs share NEUROBOLT_CAMPAIGN_DIR so subjects land in the same campaign.
#
# Slurm stdout/stderr: a new directory every submit (including resume).
# Slurm has no date token, and #SBATCH is fixed before bash runs, so this
# script mkdir's the dirs and passes --output/--error (overrides run_neuro.sh):
#   logs_baseline/YYYYMMDD-HHMMSS/output_logs/%x_%j.out
#   logs_baseline/YYYYMMDD-HHMMSS/error_logs/%x_%j.err
#   (attention → logs_attn/…). %x is the job name, %j is the Slurm job id.
# Fresh submits use the same stamp as the campaign folder. Resume uses a
# new stamp; campaign.txt in that log dir records the checkpoint campaign.
#
# Resume incomplete (no new campaign): only resubmit subjects that lack
# checkpoint-epoch*.pth under that campaign. Retries create a new per-subject
# run folder inside the same campaign (campaign folder is not recreated).
#
# Data root: prepare scans ALGERMISSEN_DATA_ROOT (same default as main.py).
# Run on the cluster where that path exists, or pass explicit sub-XXX args
# (no scan). Override: export ALGERMISSEN_DATA_ROOT=… or DATA_ROOT=…
#
# Usage:
#   bash submit_sub.sh
#   bash submit_sub.sh --batch-size 5
#   SUBJECTS_PER_JOB=5 bash submit_sub.sh
#   bash submit_sub.sh --one-per-job
#   bash submit_sub.sh --dry-run
#   bash submit_sub.sh --reuse-batches   # skip prepare if files exist
#   bash submit_sub.sh sub-001 sub-007 sub-012
#   bash submit_sub.sh --batch-size 3 sub-001 sub-002 sub-003 sub-005
#
# Resume incomplete into existing campaign (latest for mode, or explicit):
#   bash submit_sub.sh --resume-incomplete
#   bash submit_sub.sh --resume-incomplete --campaign /path/to/…/20260820-113000
#
# Attention mode (logs go to logs_attn/<stamp>/, checkpoints to log_attn/):
#   NEUROBOLT_TRAINING_MODE=attention bash submit_sub.sh
#
# Manual single batch (set campaign yourself; mkdir before sbatch):
#   STAMP=20260820-113000
#   mkdir -p logs_baseline/$STAMP/output_logs logs_baseline/$STAMP/error_logs
#   export NEUROBOLT_CAMPAIGN_DIR=…/log_baseline/$STAMP
#   sbatch --job-name=neurobolt_batch_001 \
#     --output=logs_baseline/$STAMP/output_logs/%x_%j.out \
#     --error=logs_baseline/$STAMP/error_logs/%x_%j.err \
#     run_neuro.sh subject_batches/batch_001.txt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BATCH_DIR="${BATCH_DIR:-$SCRIPT_DIR/subject_batches}"
SUBJECTS_PER_JOB="${SUBJECTS_PER_JOB:-5}"
DRY_RUN=0
REUSE_BATCHES=0
RESUME_INCOMPLETE=0
CAMPAIGN_ARG=""
EXPLICIT_SUBJECTS=()

# Keep in sync with main.py checkpoint_path (override with CHECKPOINT_PATH).
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/p_03183/personal_workspaces/sheker/NeuroBOLT/checkpoints}"

usage() {
  sed -n '2,57p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --reuse-batches) REUSE_BATCHES=1; shift ;;
    --resume-incomplete) RESUME_INCOMPLETE=1; shift ;;
    --one-per-job) SUBJECTS_PER_JOB=1; shift ;;
    --campaign)
      CAMPAIGN_ARG="${2:?--campaign requires a path}"
      shift 2
      ;;
    --campaign=*)
      CAMPAIGN_ARG="${1#*=}"
      shift
      ;;
    --batch-size)
      SUBJECTS_PER_JOB="${2:?--batch-size requires an integer}"
      shift 2
      ;;
    --batch-size=*)
      SUBJECTS_PER_JOB="${1#*=}"
      shift
      ;;
    sub-*)
      EXPLICIT_SUBJECTS+=("$1")
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

if ! [[ "$SUBJECTS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SUBJECTS_PER_JOB / --batch-size must be a positive integer (got: $SUBJECTS_PER_JOB)" >&2
  exit 1
fi

if [[ -n "$CAMPAIGN_ARG" ]] && (( ! RESUME_INCOMPLETE )); then
  echo "ERROR: --campaign is only valid with --resume-incomplete" >&2
  exit 1
fi

# Estimate: ~30 min/subject; warn if batch may exceed short 3h.
MAX_SAFE=5
if (( SUBJECTS_PER_JOB > MAX_SAFE )); then
  echo "WARNING: SUBJECTS_PER_JOB=${SUBJECTS_PER_JOB} may exceed short 3h walltime (~30 min/subject)." >&2
  echo "         Prefer --batch-size ${MAX_SAFE} or raise #SBATCH --time in run_neuro.sh." >&2
fi

MODE="${NEUROBOLT_TRAINING_MODE:-baseline}"
if [ "$MODE" = "attention" ] || [ "$MODE" = "attn" ]; then
  MODE="attention"
  LOG_FOLDER="log_attn"
else
  MODE="baseline"
  LOG_FOLDER="log_baseline"
fi
MODE_ROOT="${CHECKPOINT_PATH}/${LOG_FOLDER}"

# ---------------------------------------------------------------------------
# Helpers: campaign discovery + completed-subject detection
# ---------------------------------------------------------------------------
find_latest_campaign() {
  local mode_root="$1"
  local latest=""
  local d
  # Dated dirs: YYYYMMDD-HHMMSS
  while IFS= read -r d; do
    [ -d "$d" ] && latest="$d"
  done < <(
    ls -1d "$mode_root"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9] \
      2>/dev/null | sort
  )
  echo "$latest"
}

subject_is_complete() {
  # Match main.py subject_has_checkpoint / runtime.save_model layout.
  local campaign="$1" subject="$2"
  compgen -G "${campaign}/runs/"*"_${subject}/${subject}-VS_smooth_16s-"*"/checkpoint-epoch"*.pth \
    > /dev/null 2>&1
}

collect_subjects_from_batches() {
  local dir="$1"
  local bf line
  SUBJECT_UNIVERSE=()
  for bf in "$dir"/batch_*.txt; do
    [ -f "$bf" ] || continue
    while IFS= read -r line; do
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ "$line" == sub-* ]] || continue
      SUBJECT_UNIVERSE+=("$line")
    done < "$bf"
  done
}

write_campaign_meta() {
  local campaign="$1"
  local subjects_csv="$2"
  local resume_note="${3:-}"
  python3 - "$campaign" "$MODE" "$LOG_FOLDER" "$SUBJECTS_PER_JOB" "$subjects_csv" "$resume_note" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

campaign, mode, log_folder, batch_size, subjects_csv, resume_note = sys.argv[1:7]
subjects = [s for s in subjects_csv.split(",") if s] if subjects_csv else []
path = Path(campaign) / "campaign_meta.json"
meta = {}
if path.is_file():
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        meta = {}
if not meta:
    meta = {
        "training_mode": mode,
        "log_folder": log_folder,
        "timestamp": Path(campaign).name,
        "batch_size": int(batch_size),
        "subjects": subjects,
        "created_by": "submit_sub.sh",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
else:
    meta["batch_size"] = int(batch_size)
    if subjects:
        meta["subjects"] = subjects
if resume_note:
    resumes = meta.setdefault("resumes", [])
    resumes.append({
        "at": datetime.now(timezone.utc).isoformat(),
        "incomplete_subjects": subjects,
        "note": resume_note,
    })
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(str(path))
PY
}

# Write subject_batches/batch_NNN.txt. Explicit ids are used as given.
# With no ids, scan the Algermissen npz root and drop the same skip set as main.py.
write_subject_batches() {
  local batch_size="$1"
  local batch_dir="$2"
  shift 2
  local -a subjects=("$@")
  local data_root="${DATA_ROOT:-${ALGERMISSEN_DATA_ROOT:-/data/p_03183/data/pav_algermissen/derived/py_imported/per_block/for_neurobolt}}"
  if (( ${#subjects[@]} == 0 )); then
    if [ ! -d "$data_root" ]; then
      echo "ERROR: Algermissen data root not found: $data_root" >&2
      echo "Set ALGERMISSEN_DATA_ROOT or DATA_ROOT, or pass explicit sub-XXX args." >&2
      exit 1
    fi
    local f base id
    local -a found=()
    for f in "$data_root"/*_block*.npz; do
      [ -f "$f" ] || continue
      base="$(basename "$f")"
      id="${base%%_block*}"
      case "$id" in
        sub-004|sub-015|sub-025) continue ;;
      esac
      found+=("$id")
    done
    if (( ${#found[@]} > 0 )); then
      IFS=$'\n' subjects=($(printf '%s\n' "${found[@]}" | sort -u))
      unset IFS
    fi
  fi
  if (( ${#subjects[@]} == 0 )); then
    echo "ERROR: no subjects to batch" >&2
    exit 1
  fi
  mkdir -p "$batch_dir"
  rm -f "$batch_dir"/batch_*.txt
  local i batch_idx=0
  local -a batch
  for ((i = 0; i < ${#subjects[@]}; i += batch_size)); do
    batch_idx=$((batch_idx + 1))
    batch=("${subjects[@]:i:batch_size}")
    printf '%s\n' "${batch[@]}" > "$(printf '%s/batch_%03d.txt' "$batch_dir" "$batch_idx")"
  done
  printf '%s\n' "$batch_size" > "$batch_dir/.batch_size"
  echo "Wrote ${batch_idx} batch file(s) (${#subjects[@]} subjects) under ${batch_dir}/"
}

# ---------------------------------------------------------------------------
# Prepare batch files (full subject universe), unless resume rebuilds later
# ---------------------------------------------------------------------------
need_prepare=1
if (( REUSE_BATCHES )) && compgen -G "$BATCH_DIR"/batch_*.txt > /dev/null; then
  prev_size=""
  if [ -f "$BATCH_DIR/.batch_size" ]; then
    prev_size="$(tr -d '[:space:]' < "$BATCH_DIR/.batch_size" || true)"
  fi
  if [ -n "$prev_size" ] && [ "$prev_size" = "$SUBJECTS_PER_JOB" ] && (( ${#EXPLICIT_SUBJECTS[@]} == 0 )); then
    need_prepare=0
    echo "Reusing existing batches under ${BATCH_DIR}/ (size=${prev_size})."
  else
    echo "Regenerating batches (reuse skipped: size/explicit mismatch or missing .batch_size)."
  fi
fi

if (( need_prepare )) && (( ! RESUME_INCOMPLETE )); then
  write_subject_batches "$SUBJECTS_PER_JOB" "$BATCH_DIR" "${EXPLICIT_SUBJECTS[@]}"
elif (( RESUME_INCOMPLETE )) && (( need_prepare )) && (( ${#EXPLICIT_SUBJECTS[@]} == 0 )); then
  # Need a subject universe to filter; prepare into BATCH_DIR if missing.
  if ! compgen -G "$BATCH_DIR"/batch_*.txt > /dev/null; then
    write_subject_batches "$SUBJECTS_PER_JOB" "$BATCH_DIR"
  fi
elif (( RESUME_INCOMPLETE )) && (( ${#EXPLICIT_SUBJECTS[@]} > 0 )); then
  : # universe = explicit list below
fi

# ---------------------------------------------------------------------------
# Resolve / create campaign
# ---------------------------------------------------------------------------
SKIP_COMPLETED_EXPORT=0

if (( RESUME_INCOMPLETE )); then
  if [[ -n "$CAMPAIGN_ARG" ]]; then
    CAMPAIGN_DIR="$(cd "$CAMPAIGN_ARG" 2>/dev/null && pwd)" || {
      echo "ERROR: --campaign path not found: $CAMPAIGN_ARG" >&2
      exit 1
    }
  else
    CAMPAIGN_DIR="$(find_latest_campaign "$MODE_ROOT")"
    if [[ -z "$CAMPAIGN_DIR" ]]; then
      echo "ERROR: no campaign dirs under ${MODE_ROOT}/ (expected YYYYMMDD-HHMMSS)." >&2
      echo "       Pass --campaign PATH or run a fresh submit first." >&2
      exit 1
    fi
  fi
  echo "Campaign dir: ${CAMPAIGN_DIR}  (resume-incomplete; no new campaign)"
  # New log dir: the campaign folder is reused, this submit is a new job wave.
  LOG_TS="$(date +%Y%m%d-%H%M%S)"
  SKIP_COMPLETED_EXPORT=1

  # Subject universe: explicit args, else all ids currently in batch files.
  if (( ${#EXPLICIT_SUBJECTS[@]} > 0 )); then
    SUBJECT_UNIVERSE=("${EXPLICIT_SUBJECTS[@]}")
  else
    collect_subjects_from_batches "$BATCH_DIR"
    if (( ${#SUBJECT_UNIVERSE[@]} == 0 )); then
      echo "ERROR: no subjects found in ${BATCH_DIR}/batch_*.txt" >&2
      exit 1
    fi
  fi

  COMPLETED=()
  INCOMPLETE=()
  for s in "${SUBJECT_UNIVERSE[@]}"; do
    if subject_is_complete "$CAMPAIGN_DIR" "$s"; then
      COMPLETED+=("$s")
    else
      INCOMPLETE+=("$s")
    fi
  done

  echo "Skipping completed (${#COMPLETED[@]}): ${COMPLETED[*]:-(none)}"
  echo "Submitting incomplete (${#INCOMPLETE[@]}): ${INCOMPLETE[*]:-(none)}"

  if (( ${#INCOMPLETE[@]} == 0 )); then
    echo "Nothing to submit — all subjects in universe already have checkpoints."
    exit 0
  fi

  # Rebuild batches with incomplete only (dedicated dir; leave original batches).
  BATCH_DIR="$SCRIPT_DIR/subject_batches/resume_incomplete"
  mkdir -p "$BATCH_DIR"
  write_subject_batches "$SUBJECTS_PER_JOB" "$BATCH_DIR" "${INCOMPLETE[@]}"

  subjects_csv=$(IFS=,; echo "${INCOMPLETE[*]}")
  if (( ! DRY_RUN )); then
    write_campaign_meta "$CAMPAIGN_DIR" "$subjects_csv" "resume-incomplete"
  fi
else
  CAMPAIGN_TS="$(date +%Y%m%d-%H%M%S)"
  CAMPAIGN_DIR="${MODE_ROOT}/${CAMPAIGN_TS}"
  # Same stamp as the campaign so logs and checkpoints sort together.
  LOG_TS="$CAMPAIGN_TS"
  echo "Campaign dir: ${CAMPAIGN_DIR}  (new submit)"
  if (( ! DRY_RUN )); then
    mkdir -p "${CAMPAIGN_DIR}/runs"
  fi

  collect_subjects_from_batches "$BATCH_DIR"
  subjects_csv=$(IFS=,; echo "${SUBJECT_UNIVERSE[*]}")
  if (( ! DRY_RUN )); then
    write_campaign_meta "$CAMPAIGN_DIR" "$subjects_csv" ""
  fi
fi

BATCH_FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && BATCH_FILES+=("$line")
done < <(ls -1 "$BATCH_DIR"/batch_*.txt 2>/dev/null | sort)
if [ "${#BATCH_FILES[@]}" -eq 0 ]; then
  echo "ERROR: no batch files in $BATCH_DIR" >&2
  exit 1
fi

n_jobs=${#BATCH_FILES[@]}

echo
echo "Mode:        ${MODE}  (${LOG_FOLDER})"
if [ "$MODE" = "attention" ]; then
  SLURM_LOG_ROOT="logs_attn"
else
  SLURM_LOG_ROOT="logs_baseline"
fi
SLURM_LOG_DIR="${SLURM_LOG_ROOT}/${LOG_TS}"
SLURM_OUT_DIR="${SLURM_LOG_DIR}/output_logs"
SLURM_ERR_DIR="${SLURM_LOG_DIR}/error_logs"

echo "Campaign:    ${CAMPAIGN_DIR}"
echo "Slurm logs:  ${SLURM_OUT_DIR}/"
echo "             ${SLURM_ERR_DIR}/"
echo "Batches:     ${n_jobs}  (subjects/job target=${SUBJECTS_PER_JOB})"
echo "Walltime:    ~$(( SUBJECTS_PER_JOB * 30 )) min/job budget (assume ~30 min/subject; short=180 min)"
if (( SKIP_COMPLETED_EXPORT )); then
  echo "Skip done:   NEUROBOLT_SKIP_COMPLETED=1 (main.py also skips completed in campaign)"
fi
echo

if (( ! DRY_RUN )); then
  mkdir -p "$SLURM_OUT_DIR" "$SLURM_ERR_DIR"
  {
    echo "campaign=${CAMPAIGN_DIR}"
    echo "training_mode=${MODE}"
    echo "submitted_at=${LOG_TS}"
  } > "${SLURM_LOG_DIR}/campaign.txt"
fi

# Pass batch path as $1 to run_neuro.sh (SLURM forwards args after the script).
# Job name via CLI — #SBATCH --job-name=neurobolt_$VAR does not expand at submit.
# Relative path for resume batches under subject_batches/resume_incomplete/.
job_idx=0
for batchfile in "${BATCH_FILES[@]}"; do
  job_idx=$((job_idx + 1))
  batch_base="$(basename "$batchfile" .txt)"   # e.g. batch_001
  # Prefer path relative to code root so the job finds it after cd.
  if [[ "$batchfile" == "$SCRIPT_DIR"/* ]]; then
    rel_batch="${batchfile#"$SCRIPT_DIR"/}"
  else
    rel_batch="subject_batches/$(basename "$batchfile")"
  fi
  n_in_batch="$(grep -cE '^[[:space:]]*sub-' "$batchfile" || true)"
  subjects_preview="$(tr '\n' ',' < "$batchfile" | sed 's/,$//')"

  echo "[${job_idx}/${n_jobs}] ${batch_base}  (${n_in_batch}): ${subjects_preview}"
  if (( DRY_RUN )); then
    echo "  (dry-run) NEUROBOLT_CAMPAIGN_DIR=${CAMPAIGN_DIR} \\"
    echo "            NEUROBOLT_TRAINING_MODE=${MODE} \\"
    if (( SKIP_COMPLETED_EXPORT )); then
      echo "            NEUROBOLT_SKIP_COMPLETED=1 \\"
    fi
    echo "            sbatch --job-name=neurobolt_${batch_base} \\"
    echo "              --output=${SLURM_OUT_DIR}/%x_%j.out \\"
    echo "              --error=${SLURM_ERR_DIR}/%x_%j.err \\"
    echo "              --export=ALL run_neuro.sh ${rel_batch}"
  else
    export NEUROBOLT_CAMPAIGN_DIR="$CAMPAIGN_DIR"
    export NEUROBOLT_TRAINING_MODE="$MODE"
    if (( SKIP_COMPLETED_EXPORT )); then
      export NEUROBOLT_SKIP_COMPLETED=1
    else
      unset NEUROBOLT_SKIP_COMPLETED || true
    fi
    sbatch \
      --job-name="neurobolt_${batch_base}" \
      --output="${SLURM_OUT_DIR}/%x_%j.out" \
      --error="${SLURM_ERR_DIR}/%x_%j.err" \
      --export=ALL \
      "$SCRIPT_DIR/run_neuro.sh" \
      "$rel_batch"
  fi
done

echo
if (( DRY_RUN )); then
  echo "Dry-run only — no jobs submitted."
  echo "Would use campaign: ${CAMPAIGN_DIR}"
else
  echo "Done. Campaign dir: ${CAMPAIGN_DIR}"
  echo "Check with: squeue -u \$USER"
fi
