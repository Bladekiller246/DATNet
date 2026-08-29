<#
.SYNOPSIS
    Wait for a training run to reach its target iteration, then build the report.

.DESCRIPTION
    Polls the run's ledger. When the run reports `complete` -- or the target
    iteration is reached and no trainer process is left alive -- it runs
    scripts/report_phase.py once and exits.

    It NEVER starts training. It only turns a finished run into numbers, a gate
    figure and sample images, so there is something to look at in the morning.

    If the run dies and its watchdog cannot recover it, this exits after
    -MaxHours without producing a report, rather than waiting forever.

.EXAMPLE
    .\scripts\watch_and_report.ps1 -RunDir runs\phase1_denoise__dual__seed0 -TargetIters 60000
#>
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][int]$TargetIters,
    [int]$PollSeconds = 120,
    [int]$EmptyPollsBeforeGivingUp = 5,
    [double]$MaxHours = 14,
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$ledger = Join-Path $RunDir "ledger.jsonl"
$deadline = (Get-Date).AddHours($MaxHours)
$log = Join-Path $RunDir "watch_and_report.log"

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

$emptyPolls = 0
Note "watching $RunDir for iteration $TargetIters"

while ($true) {
    if ((Get-Date) -gt $deadline) {
        Note "gave up after $MaxHours h without the run completing. No report written."
        exit 1
    }

    $iter = -1
    if (Test-Path $ledger) {
        # last segment_end record carries the authoritative iteration
        $ends = Select-String -Path $ledger -Pattern '"kind": "segment_end"' -ErrorAction SilentlyContinue
        if ($ends) {
            $last = $ends[-1].Line | ConvertFrom-Json
            $iter = $last.iteration
        }
    }

    $trainers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                  Where-Object { $_.CommandLine -like '*train.py*' })

    if ($iter -ge $TargetIters) {
        Note "run reached $iter / $TargetIters -- building report"
        break
    }
    if ($trainers.Count -eq 0 -and $iter -gt 0) {
        # A missing trainer is NOT proof the run stopped: run_segments.ps1 leaves
        # a cooldown gap between segments with no train.py alive. Reporting on
        # the first empty poll fired this watcher at iteration 32,000 of 60,000.
        # Require the gap to persist well beyond the cooldown before believing it.
        $emptyPolls++
        if ($emptyPolls -ge $EmptyPollsBeforeGivingUp) {
            Note "no trainer for $emptyPolls consecutive polls at $iter / $TargetIters -- run really stopped; reporting"
            break
        }
        Note "no trainer alive (poll $emptyPolls/$EmptyPollsBeforeGivingUp) -- probably an inter-segment cooldown; waiting"
    }
    else { $emptyPolls = 0 }

    Start-Sleep -Seconds $PollSeconds
}

Note "running report_phase.py"
& $Python "scripts/report_phase.py" --run-dir $RunDir
Note "report finished with exit code $LASTEXITCODE"
