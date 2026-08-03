#!/usr/bin/env bash
# Reproduce Figure 8 end-to-end: prep_data -> train -> backtest for all 5 variants.
#
# Runtime (single RTX-class GPU, 60k timesteps x 5 agents x 5 variants): ~4-6 h.
# Runtime on CPU: multiply by 3-5x. The DES75/DES65/DES60 variants share the
# same 8 indicators + cov backbone; they differ only in which DES accuracy
# column is joined in during prep, so their train wall-times are near-identical.
#
# Seeding:
#   Every train/backtest script now imports finrl_seeds and calls
#   set_finrl_seeds(read_seed_from_env(default=42)) as its first non-import
#   statement. Override globally with:
#
#       export FINRL_SEED=17
#
# The five variants produced (matching paper Figure 8 legend):
#   1. baseline                (StockTradingEnv, 8 indicators, no cap anchor)
#   2. capweighted_finrl       (CapAnchoredPortfolioEnv, 8 indicators)
#   3. capweighted_finrl_des75 (2 + DES @ 75 % accuracy)
#   4. capweighted_finrl_des65 (2 + DES @ 65 %)
#   5. capweighted_finrl_des60 (2 + DES @ 60 %)
#
# Nothing is retrained if the target pickle / .zip already exists; delete the
# artefacts under finrl/ to force a rerun. Pass --force to skip that check.
#
# Usage:
#   bash finrl/backtest/run_all.sh                 # all 5 variants
#   bash finrl/backtest/run_all.sh baseline des75  # subset by nickname
#   FINRL_SEED=7 bash finrl/backtest/run_all.sh --force
set -euo pipefail

# Resolve finrl/ regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINRL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FINRL_DIR}"

FORCE=0
declare -a REQUESTED=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,32p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) REQUESTED+=("$arg") ;;
    esac
done

# Variant table: nickname | prep script | train script | backtest script | pickle prefix | model dir
declare -a VARIANTS=(
    "baseline|prep_data/tw50_2024_20260331_prep_data.py|train/tw50_2024_20260331_train.py|backtest/tw50_2024_20260331_backtest.py|tw50_2024_20260331|tw50_2024_20260331_trained_models"
    "capweighted_finrl|prep_data/tw50_capweighted_finrl_prep.py|train/tw50_capweighted_finrl_train.py|backtest/tw50_capweighted_finrl_backtest.py|tw50_capweighted_finrl|tw50_capweighted_finrl_trained_models"
    "des75|prep_data/tw50_capweighted_finrl_des_prep.py|train/tw50_capweighted_finrl_des_train.py|backtest/tw50_capweighted_finrl_des_backtest.py|tw50_capweighted_finrl_des|tw50_capweighted_finrl_des_trained_models"
    "des65|prep_data/tw50_capweighted_finrl_des65_prep.py|train/tw50_capweighted_finrl_des65_train.py|backtest/tw50_capweighted_finrl_des65_backtest.py|tw50_capweighted_finrl_des65|tw50_capweighted_finrl_des65_trained_models"
    "des60|prep_data/tw50_capweighted_finrl_des60_prep.py|train/tw50_capweighted_finrl_des60_train.py|backtest/tw50_capweighted_finrl_des60_backtest.py|tw50_capweighted_finrl_des60|tw50_capweighted_finrl_des60_trained_models"
)

want() {
    if [ ${#REQUESTED[@]} -eq 0 ]; then return 0; fi
    for r in "${REQUESTED[@]}"; do
        [ "$r" = "$1" ] && return 0
    done
    return 1
}

run_variant() {
    local nick="$1" prep="$2" train="$3" bt="$4" pkl="$5" mdl="$6"
    echo ""
    echo "============================================================"
    echo "[${nick}] prep -> train -> backtest"
    echo "============================================================"

    if [ "$FORCE" -eq 1 ] || [ ! -f "${pkl}_train.pkl" ] || [ ! -f "${pkl}_trade.pkl" ]; then
        echo "[${nick}] prep: python ${prep}"
        python "${prep}"
    else
        echo "[${nick}] prep: pickles exist, skipping (use --force to rebuild)"
    fi

    if [ "$FORCE" -eq 1 ] || [ ! -f "${mdl}/agent_sac.zip" ]; then
        echo "[${nick}] train: python ${train}"
        python "${train}"
    else
        echo "[${nick}] train: agent_sac.zip exists, skipping (use --force to retrain)"
    fi

    echo "[${nick}] backtest: python ${bt}"
    python "${bt}"
}

for row in "${VARIANTS[@]}"; do
    IFS='|' read -r nick prep train bt pkl mdl <<< "$row"
    if want "$nick"; then
        run_variant "$nick" "$prep" "$train" "$bt" "$pkl" "$mdl"
    else
        echo "[${nick}] SKIPPED (not in requested subset)"
    fi
done

echo ""
echo "All requested variants finished. Figure 8 inputs are under:"
echo "  finrl/results/baseline/"
echo "  finrl/results/capweighted_finrl/"
echo "  finrl/results/capweighted_finrl_des75/"
echo "  finrl/results/capweighted_finrl_des65/"
echo "  finrl/results/capweighted_finrl_des60/"
