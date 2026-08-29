<#
.SYNOPSIS
    Wait for every arm of a phase to finish, then produce the verdict.

.DESCRIPTION
    Polls each arm's ledger. When all of them have reached -TargetIters -- or the
    queue has clearly stopped -- it runs report_phase.py per arm and then
    compare_arms.py, which is the actual Phase-1 deliverable: does `dual` beat
    both single-axis arms at matched parameters.

    Starts no training. If an arm diverged (exit 3) the queue will have stopped
    early; this still reports on whatever completed, so a partial night is not a
    wasted night.

.EXAMPLE
    .\scripts\watch_phase.ps1 -Phase phase1_denoise -TargetIters 30000
#>
param(
    [Parameter(Mandatory = $true)][string]$Phase,
    [Parameter(Mandatory = $true)][int]$TargetIters,
    [string[]]$Arms = @("dual", "channel_only", "window_only"),
    [int]$Seed = 0,
    [string]$RunRoot = "runs",
    [int]$PollSeconds = 180,
    [int]$EmptyPollsBeforeGivingUp = 6,
    [double]$MaxHours = 20,
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$log = Join-Path $RunRoot "$Phase`_watch.log"
$deadline = (Get-Date).AddHours($MaxHours)
$emptyPolls = 0

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function ArmIteration($arm) {
    $ledger = Join-Path $RunRoot "$Phase`__$arm`__seed$Seed/ledger.jsonl"
    if (-not (Test-Path $ledger)) { return -1 }
    $ends = Select-String -Path $ledger -Pattern '"kind": "segment_end"' -ErrorAction SilentlyContinue
    if (-not $ends) { return 0 }
    return ($ends[-1].Line | ConvertFrom-Json).iteration
}

Note "watching $Phase for $($Arms.Count) arms x $TargetIters iterations"

while ($true) {
    if ((Get-Date) -gt $deadline) {
        Note "gave up after $MaxHours h. Reporting on whatever finished."
        break
    }

    $done = 0
    $status = @()
    foreach ($a in $Arms) {
        $i = ArmIteration $a
        $status += "$a=$i"
        if ($i -ge $TargetIters) { $done++ }
    }

    if ($done -eq $Arms.Count) {
        Note "all arms complete ($($status -join ', ')) -- reporting"
        break
    }

    $alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -like '*train.py*' })
    if ($alive.Count -eq 0) {
        # cooldown gaps and the handover between arms both leave no trainer
        # running, so one empty poll proves nothing
        $emptyPolls++
        if ($emptyPolls -ge $EmptyPollsBeforeGivingUp) {
            Note "no trainer for $emptyPolls polls ($($status -join ', ')) -- queue stopped; reporting"
            break
        }
        Note "no trainer (poll $emptyPolls/$EmptyPollsBeforeGivingUp) -- $($status -join ', ')"
    }
    else {
        $emptyPolls = 0
        Note "training: $($status -join ', ')"
    }

    Start-Sleep -Seconds $PollSeconds
}

foreach ($a in $Arms) {
    $dir = Join-Path $RunRoot "$Phase`__$a`__seed$Seed"
    if (Test-Path $dir) {
        Note "report_phase.py $a"
        & $Python "scripts/report_phase.py" --run-dir $dir
    }
}

Note "compare_arms.py -- the Phase verdict"
& $Python "scripts/compare_arms.py" --phase $Phase --seed $Seed
Note "done (exit $LASTEXITCODE)"
