<#
.SYNOPSIS
    Run ONE training run to completion as a sequence of bounded segments,
    restarting automatically if it crashes.

.DESCRIPTION
    Calls scripts/train.py repeatedly. Each call trains at most -SegmentIters
    iterations and exits; exit code 10 means "more to do", 0 means complete.

    WATCHDOG. If train.py dies for any other reason -- OOM, driver reset, a
    transient CUDA fault, the machine waking from sleep -- the run is restarted
    from its last checkpoint after -RetryDelaySeconds, up to -MaxRetries
    consecutive failures. Any successful segment resets the failure counter, so
    a long run survives repeated unrelated hiccups. Progress is never lost
    beyond the last checkpoint because train.py resumes exactly, RNG included.

    ONE RUN ONLY. This script deliberately refuses to advance to a different
    variant on its own -- finishing an arm and silently starting the next one
    spends hours of GPU time on work that may not be wanted once the first
    result is visible. Pass -AllowMultipleVariants to override, and only when
    explicitly asked for.

    DISK WATCHDOG. Before every launch, checks free space on the repo's drive
    against -MinFreeGB. Below that, it stops immediately (exit 4) instead of
    retrying -- a full disk fails the same way every time, so retrying just
    burns the -MaxRetries budget on a problem retrying cannot fix.

    CRASH-CAUSE WATCHDOG. On an unrecognised exit code, best-effort peeks at
    the run's ledger.jsonl for an OOM-backoff record at micro-batch 1 right
    before the crash, and names that in the retry message instead of leaving
    it an anonymous exit code.

    Stop it any time with Ctrl-C. Re-run the same command to continue.

.EXAMPLE
    .\scripts\run_segments.ps1 -Config configs\phase1_denoise.yaml -Variant dual
#>
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string[]]$Variant = @("dual"),
    [int]$SegmentIters = 10000,
    [double]$MaxHours = 0,          # 0 = no limit
    [int]$CoolDownSeconds = 45,
    [int]$MaxRetries = 5,
    [int]$RetryDelaySeconds = 60,
    [int]$Seed = 0,
    [double]$MinFreeGB = 5,
    [switch]$AllowMultipleVariants,
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# --- Disk-space watchdog ---------------------------------------------------
# Checkpoints, samples and the report figures all land under runs\ / results\
# on the same drive as the repo. A segment that dies mid-write because the
# drive filled is a worse failure than any crash the retry loop already
# handles: _save() in trainer.py refuses to write non-finite weights, but it
# cannot refuse to write to a full disk. Catch it before launching, not after.
function Test-DiskWatchdog {
    param([string]$Path, [double]$MinGB)
    $resolvedRoot = (Resolve-Path $Path).Path
    $driveLetter = (Get-Item $resolvedRoot).PSDrive.Name
    $drive = Get-PSDrive -Name $driveLetter
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    return @{ FreeGB = $freeGB; Ok = ($freeGB -ge $MinGB) }
}

# --- Crash-cause watchdog ---------------------------------------------------
# Best-effort: an unrecognised exit code is retried blindly today. If the
# run's own ledger shows OOM backoff already collapsed the micro-batch to 1
# right before the crash, that is worth saying out loud in the retry message
# instead of leaving it as an anonymous "exit 1" -- it tells a human reading
# the log the next morning that a smaller model/batch is the fix, not luck.
function Get-CrashCauseWatchdog {
    param([string]$Config, [string]$Variant, [int]$Seed)
    try {
        $nameLine = Select-String -Path $Config -Pattern '^name:\s*(\S+)' -ErrorAction SilentlyContinue
        if (-not $nameLine) { return "unknown" }
        $cfgName = $nameLine.Matches[0].Groups[1].Value
        $runDir = Join-Path "runs" "${cfgName}__${Variant}__seed${Seed}"
        $ledger = Join-Path $runDir "ledger.jsonl"
        if (-not (Test-Path $ledger)) { return "unknown" }
        $tail = Get-Content -Path $ledger -Tail 3 -ErrorAction SilentlyContinue
        foreach ($line in $tail) {
            $rec = $null
            try { $rec = $line | ConvertFrom-Json -ErrorAction Stop } catch { continue }
            if ($rec.kind -eq "oom_backoff" -and $rec.micro_batch -le 1) {
                return "oom (micro-batch already at 1)"
            }
        }
    } catch { }
    return "unknown"
}

if ($Variant.Count -gt 1 -and -not $AllowMultipleVariants) {
    Write-Host "Refusing to run $($Variant.Count) variants in one queue." -ForegroundColor Red
    Write-Host "Chaining runs is opt-in: pass -AllowMultipleVariants if that is really wanted." -ForegroundColor Red
    Write-Host "Otherwise launch one arm, let it finish, then decide." -ForegroundColor Yellow
    exit 2
}

$deadline = $null
if ($MaxHours -gt 0) { $deadline = (Get-Date).AddHours($MaxHours) }

function Write-Stamp($msg, $colour) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) -ForegroundColor $colour
}

foreach ($v in $Variant) {
    Write-Host ""
    Write-Stamp "=== $Config / $v ===" "Cyan"
    $failures = 0

    while ($true) {
        if ($deadline -ne $null -and (Get-Date) -gt $deadline) {
            Write-Stamp "time budget reached; stopping. Re-run to continue." "Yellow"
            exit 0
        }

        $disk = Test-DiskWatchdog -Path $root -MinGB $MinFreeGB
        if (-not $disk.Ok) {
            Write-Stamp "DISK WATCHDOG -- $($disk.FreeGB) GB free, below -MinFreeGB $MinFreeGB." "Red"
            Write-Stamp "Not launching: retrying would only fail the same way. Free space and re-run." "Red"
            exit 4
        }

        & $Python "scripts/train.py" --config $Config --variant $v `
            --segment-iters $SegmentIters --seed $Seed
        $code = $LASTEXITCODE

        if ($code -eq 0) {
            Write-Stamp "$v COMPLETE." "Green"
            break
        }
        elseif ($code -eq 10) {
            $failures = 0     # a good segment clears the failure streak
            if ($CoolDownSeconds -gt 0) {
                Write-Stamp "segment done; cooling $CoolDownSeconds s" "DarkGray"
                Start-Sleep -Seconds $CoolDownSeconds
            }
        }
        elseif ($code -eq 3) {
            Write-Stamp "DIVERGED -- non-finite gradients. NOT retrying: resuming would" "Red"
            Write-Stamp "reload the same dead state. Lower the LR and restart from a good checkpoint." "Red"
            exit 3
        }
        else {
            $failures++
            $cause = Get-CrashCauseWatchdog -Config $Config -Variant $v -Seed $Seed
            if ($failures -gt $MaxRetries) {
                Write-Stamp "train.py failed $failures times in a row (exit $code, cause: $cause). Giving up." "Red"
                exit $code
            }
            Write-Stamp "train.py exited $code (cause: $cause) -- retry $failures/$MaxRetries in $RetryDelaySeconds s (resumes from last checkpoint)" "Yellow"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
}

Write-Host ""
Write-Stamp "done. Nothing further will start automatically." "Green"
