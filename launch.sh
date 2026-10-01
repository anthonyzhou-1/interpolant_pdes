#!/bin/bash
#
# Submit the experiment grid to Slurm. Prints the sbatch commands unless --submit is given.
#
#   ./launch.sh train                                   # 3 datasets x 5 models x 3 seeds
#   ./launch.sh train --datasets km_flow --dt 4,8 --seeds 42
#   ./launch.sh eval  --out /path/to/data/nfe_results   # NFE sweeps of the trained models
#   ./launch.sh crps                                    # climate CRPS/SSR vs lead time
#   ./launch.sh bias                                    # climate 10-year bias rollouts
#
# Options: --datasets a,b  --models a,b  --seeds a,b  --dt a,b  --nfes 2:5:10:20
#          --ensemble E  --ckpt-root DIR  --out DIR  --submit
# Extra sbatch flags (partition, constraint, ...) can be passed via $SBATCH_ARGS.
# The current Python environment is inherited by the job.

set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:-}"; shift || true
DATASETS="km_flow,rayleigh_benard,climate"
MODELS=""
SEEDS="42,43,44"
DT=""
NFES="2:5:10:20"
ENSEMBLE=8
CKPT_ROOT="/path/to/data/logs"
OUT_DIR=""
SUBMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)  DATASETS="$2"; shift 2 ;;
        --models)    MODELS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --dt)        DT="$2"; shift 2 ;;
        --nfes)      NFES="$2"; shift 2 ;;
        --ensemble)  ENSEMBLE="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --out)       OUT_DIR="$2"; shift 2 ;;
        --submit)    SUBMIT=1; shift ;;
        *) sed -n '2,15p' "$0"; exit 1 ;;
    esac
done
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"
if [[ -n "$DT" ]]; then IFS=',' read -ra DT_ARR <<< "$DT"; else DT_ARR=(""); fi
mkdir -p logs/slurm

# ddpm, ddim and tsm share one trained model (trained as ddim) and differ only in sampling.
models_for() {
    local base="fno2d"; [[ "$1" == climate ]] && base="sfno"
    if [[ "$2" == train ]]; then echo "$base ddim edm flow_matching interpolant"
    else echo "$base ddpm ddim tsm edm flow_matching interpolant"; fi
}
train_model_for() { case "$1" in ddpm|ddim|tsm) echo ddim ;; *) echo "$1" ;; esac; }
# Checkpoint read for evaluation: a fixed step all runs reached, or the best on km_flow.
ckpt_for() {
    case "$1" in
        rayleigh_benard) echo "model_step_00060000.ckpt" ;;
        climate)         echo "model_step_00050000.ckpt" ;;
        *)               echo "*_best.ckpt" ;;
    esac
}
time_for() {
    case "$2:$1" in
        train:climate) echo "16:00:00" ;;
        *:km_flow)     echo "6:00:00" ;;
        *)             echo "12:00:00" ;;
    esac
}
val_bs_for() { case "$1" in km_flow) echo 8 ;; rayleigh_benard) echo 4 ;; climate) echo 2 ;; esac; }

# Newest run dir for (model, dataset, description, seed) containing the requested checkpoint.
find_ckpt() {
    local d c
    for d in $(ls -td "${CKPT_ROOT}"/${1}_${2}_${3}_${4}_* 2>/dev/null || true); do
        c="$(ls -t "${d}"/$5 2>/dev/null | head -1 || true)"
        [[ -n "$c" ]] && { echo "$c"; return; }
    done
    echo "  !! no $5 for ${1}_${2}_${3}_${4}_* in ${CKPT_ROOT} -- skipping" >&2
}

n=0
submit() {  # submit <job-name> <time> <command...>
    local name="$1" time="$2"; shift 2
    local cmd=(sbatch --job-name="$name" --time="$time" -N 1 --gpus-per-node=1 --cpus-per-task=8
               --mem=128G -o "logs/slurm/${name}.%j.out" ${SBATCH_ARGS:-} --wrap="$*")
    n=$((n+1))
    if [[ "$SUBMIT" == 1 ]]; then "${cmd[@]}"; else printf '%q ' "${cmd[@]}"; echo; fi
}

case "$STAGE" in
train)
    for ds in "${DS_ARR[@]}"; do
        read -ra M_ARR <<< "${MODELS//,/ }"; [[ -z "$MODELS" ]] && read -ra M_ARR <<< "$(models_for "$ds" train)"
        for dt in "${DT_ARR[@]}"; do
            [[ -n "$dt" && "$ds" == climate ]] && continue   # climate has a fixed 6 h step
            for model in "${M_ARR[@]}"; do for seed in "${SEED_ARR[@]}"; do
                args="--config=configs/${ds}.yaml --model_name=${model} --seed=${seed}"
                [[ -n "$dt" ]] && args+=" --dt_stride=${dt} --description=dt${dt}"
                submit "tr_${ds}_${model}_${seed}${dt:+_dt$dt}" "$(time_for "$ds" train)" python train.py $args
            done; done
        done
    done ;;
eval)
    OUT_DIR="${OUT_DIR:-${CKPT_ROOT}/nfe_results}"
    for ds in "${DS_ARR[@]}"; do
        read -ra M_ARR <<< "${MODELS//,/ }"; [[ -z "$MODELS" ]] && read -ra M_ARR <<< "$(models_for "$ds" eval)"
        for dt in "${DT_ARR[@]}"; do
            [[ -n "$dt" && "$ds" == climate ]] && continue
            out="$OUT_DIR"; [[ -n "$dt" ]] && out="${OUT_DIR%/}_dt${dt}"
            for model in "${M_ARR[@]}"; do for seed in "${SEED_ARR[@]}"; do
                ckpt="$(find_ckpt "$(train_model_for "$model")" "$ds" "${dt:+dt$dt}" "$seed" "$(ckpt_for "$ds")")"
                [[ -z "$ckpt" ]] && continue
                case "$model" in
                    fno2d|sfno) ens=1; nfes=1 ;;
                    ddpm)       ens="$ENSEMBLE"; nfes=1 ;;
                    *)          ens="$ENSEMBLE"; nfes="${NFES//:/ }" ;;
                esac
                case "$model" in edm|interpolant) variants="ode sde" ;; *) variants="default" ;; esac
                for var in $variants; do
                    args="--config=configs/${ds}.yaml --model_name=${model} --seed=${seed} --checkpoint=${ckpt}"
                    args+=" --out=${out}/E${ens} --nfes ${nfes} --ensemble ${ens} --num_val_samples=64"
                    [[ "$ens" != 1 ]] && args+=" --val_batch_size=$(val_bs_for "$ds")"
                    [[ "$var" != default ]] && args+=" --variants ${var}"
                    [[ -n "$dt" ]] && args+=" --dt_stride=${dt} --description=dt${dt}"
                    submit "ev_${ds}_${model}_${var}_${seed}${dt:+_dt$dt}" "$(time_for "$ds" eval)" python eval_nfe.py $args
                done
            done; done
        done
    done ;;
crps|bias)
    # model : variant : sigma -- the climate samplers compared at 10 NFE (20 for SDEs in crps)
    RUNS=("ddim::" "edm:ode:" "edm:sde:" "flow_matching::" "interpolant:ode:" "interpolant:sde:0.5")
    for run in "${RUNS[@]}"; do
        IFS=':' read -r model var sigma <<< "$run"
        [[ -n "$MODELS" && ",$MODELS," != *",$model,"* ]] && continue
        for seed in "${SEED_ARR[@]}"; do
            ckpt="$(find_ckpt "$model" climate "" "$seed" "$(ckpt_for climate)")"
            [[ -z "$ckpt" ]] && continue
            args="--model_name=${model} --seed=${seed} --checkpoint=${ckpt}"
            [[ -n "$sigma" ]] && args+=" --sigma=${sigma}"
            if [[ "$STAGE" == crps ]]; then
                [[ "$model" == interpolant && "$var" == ode ]] && continue
                nfe=10; [[ "$var" == sde ]] && nfe=20
                cvar="$var"; [[ "$model" == edm ]] && cvar="${var}-euler"
                args+=" --nfe=${nfe} --ensemble=16 --out=${OUT_DIR:-${CKPT_ROOT}/crps_results}"
                [[ -n "$cvar" ]] && args+=" --variant=${cvar}"
                submit "crps_${model}_${cvar:-default}_${seed}" "12:00:00" python eval_crps_leadtime.py $args
            else
                args+=" --nfe=10 --out=${OUT_DIR:-${CKPT_ROOT}/bias_results}"
                [[ -n "$var" ]] && args+=" --variant=${var}"
                submit "bias_${model}_${var:-default}_${seed}" "12:00:00" python eval_bias_10yr.py $args
            fi
        done
    done ;;
*)
    sed -n '2,15p' "$0"; exit 1 ;;
esac

echo "--- ${n} job(s)$([[ $SUBMIT == 1 ]] && echo ' submitted' || echo ' (dry run; add --submit)')"
