<#
.SYNOPSIS
Reproduce Figure 8 end-to-end (Windows PowerShell twin of run_all.sh).

.DESCRIPTION
Runs prep -> train -> backtest for each of the five FinRL variants:
  baseline, capweighted_finrl, des75, des65, des60.

Every train/backtest script imports finrl_seeds and seeds Python/NumPy/PyTorch/SB3
with $env:FINRL_SEED (default 42) before constructing the environment.

Runtime: ~4-6 h on a single RTX-class GPU, longer on CPU.

.PARAMETER Variants
Optional subset of nicknames to run. Default: all 5.

.PARAMETER Force
Rebuild pickles and retrain even if artifacts already exist.

.EXAMPLE
    .\finrl\backtest\run_all.ps1

.EXAMPLE
    $env:FINRL_SEED = "7"
    .\finrl\backtest\run_all.ps1 -Variants baseline,des75 -Force
#>
[CmdletBinding()]
param(
    [string[]] $Variants = @(),
    [switch]   $Force
)

$ErrorActionPreference = "Stop"

# Resolve finrl/ regardless of invocation directory.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FinrlDir  = Resolve-Path (Join-Path $ScriptDir "..")
Push-Location $FinrlDir

$VariantTable = @(
    @{ nick = "baseline";          prep = "prep_data/tw50_2024_20260331_prep_data.py";      train = "train/tw50_2024_20260331_train.py";              bt = "backtest/tw50_2024_20260331_backtest.py";              pkl = "tw50_2024_20260331";              mdl = "tw50_2024_20260331_trained_models" }
    @{ nick = "capweighted_finrl"; prep = "prep_data/tw50_capweighted_finrl_prep.py";       train = "train/tw50_capweighted_finrl_train.py";          bt = "backtest/tw50_capweighted_finrl_backtest.py";          pkl = "tw50_capweighted_finrl";          mdl = "tw50_capweighted_finrl_trained_models" }
    @{ nick = "des75";             prep = "prep_data/tw50_capweighted_finrl_des_prep.py";   train = "train/tw50_capweighted_finrl_des_train.py";      bt = "backtest/tw50_capweighted_finrl_des_backtest.py";      pkl = "tw50_capweighted_finrl_des";      mdl = "tw50_capweighted_finrl_des_trained_models" }
    @{ nick = "des65";             prep = "prep_data/tw50_capweighted_finrl_des65_prep.py"; train = "train/tw50_capweighted_finrl_des65_train.py";    bt = "backtest/tw50_capweighted_finrl_des65_backtest.py";    pkl = "tw50_capweighted_finrl_des65";    mdl = "tw50_capweighted_finrl_des65_trained_models" }
    @{ nick = "des60";             prep = "prep_data/tw50_capweighted_finrl_des60_prep.py"; train = "train/tw50_capweighted_finrl_des60_train.py";    bt = "backtest/tw50_capweighted_finrl_des60_backtest.py";    pkl = "tw50_capweighted_finrl_des60";    mdl = "tw50_capweighted_finrl_des60_trained_models" }
)

function Should-Run($nick) {
    if ($Variants.Count -eq 0) { return $true }
    return $Variants -contains $nick
}

foreach ($v in $VariantTable) {
    if (-not (Should-Run $v.nick)) {
        Write-Host "[$($v.nick)] SKIPPED (not in requested subset)"
        continue
    }
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$($v.nick)] prep -> train -> backtest"
    Write-Host "============================================================"

    $trainPkl = "$($v.pkl)_train.pkl"
    $tradePkl = "$($v.pkl)_trade.pkl"
    if ($Force -or -not (Test-Path $trainPkl) -or -not (Test-Path $tradePkl)) {
        Write-Host "[$($v.nick)] prep: python $($v.prep)"
        python $v.prep
    } else {
        Write-Host "[$($v.nick)] prep: pickles exist, skipping (use -Force to rebuild)"
    }

    $sacZip = Join-Path $v.mdl "agent_sac.zip"
    if ($Force -or -not (Test-Path $sacZip)) {
        Write-Host "[$($v.nick)] train: python $($v.train)"
        python $v.train
    } else {
        Write-Host "[$($v.nick)] train: agent_sac.zip exists, skipping (use -Force to retrain)"
    }

    Write-Host "[$($v.nick)] backtest: python $($v.bt)"
    python $v.bt
}

Pop-Location

Write-Host ""
Write-Host "All requested variants finished. Figure 8 inputs are under:"
Write-Host "  finrl/results/baseline/"
Write-Host "  finrl/results/capweighted_finrl/"
Write-Host "  finrl/results/capweighted_finrl_des75/"
Write-Host "  finrl/results/capweighted_finrl_des65/"
Write-Host "  finrl/results/capweighted_finrl_des60/"
