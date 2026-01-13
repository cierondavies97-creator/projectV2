[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]  [string]$SnapshotId,
  [Parameter(Mandatory=$true)]  [string]$RunId,
  [Parameter(Mandatory=$true)]  [string]$TradingDay,   # "YYYY-MM-DD"
  [Parameter(Mandatory=$true)]  [string]$ClusterId,

  [ValidateSet("research","backtest","paper","live","dev","prod")]
  [string]$Env = "research",

  [ValidateSet("backtest","paper","live")]
  [string]$Mode = "backtest",

  [int]$BaseSeed = 0,

  # If you use a venv, set e.g. ".\.venv\Scripts\python.exe"
  [string]$PythonExe = "python",

  # Some scripts use --snapshot_id instead of --snapshot-id.
  [ValidateSet("kebab","underscore")]
  [string]$ArgStyle = "kebab",

  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..")).Path

Push-Location $RepoRoot
try {
  # Ensure imports resolve from src/
  $env:PYTHONPATH = (Join-Path $RepoRoot "src")

  $scriptPath = (Join-Path $RepoRoot "scripts\run_microbatch.py")

  if (-not (Test-Path $scriptPath)) {
    throw "Cannot find $scriptPath. Confirm repo root and that scripts\run_microbatch.py exists."
  }

  # Arg naming style toggle (kebab-case vs underscore)
  if ($ArgStyle -eq "kebab") {
    $args = @(
      $scriptPath,
      "--env", $Env,
      "--mode", $Mode,
      "--snapshot-id", $SnapshotId,
      "--run-id", $RunId,
      "--trading-day", $TradingDay,
      "--cluster-id", $ClusterId,
      "--base-seed", "$BaseSeed"
    )
  } else {
    $args = @(
      $scriptPath,
      "--env", $Env,
      "--mode", $Mode,
      "--snapshot_id", $SnapshotId,
      "--run_id", $RunId,
      "--trading_day", $TradingDay,
      "--cluster_id", $ClusterId,
      "--base_seed", "$BaseSeed"
    )
  }

  Write-Host "RepoRoot:  $RepoRoot"
  Write-Host "PYTHONPATH: $env:PYTHONPATH"
  Write-Host "Command:   $PythonExe $($args -join ' ')"

  if (-not $DryRun) {
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
      throw "run_microbatch failed with exit code $LASTEXITCODE"
    }
  }
}
finally {
  Pop-Location
}
