$ErrorActionPreference = "Stop"

function Assert-Ok([string]$label) {
  if ($LASTEXITCODE -ne 0) { throw "$label failed (exit=$LASTEXITCODE)" }
}

# --- params (edit if needed) ---
$envName  = "research"
$mode     = "backtest"
$snapshot = "dev_retail_v1"
$cluster  = "metals"
$baseSeed = 0

$inst     = "XAUUSD"
$dt       = "2019-03-08"
$runId    = "smoke_$($dt.Replace('-',''))_metals"

$env:PYTHONPATH=".\src"

Write-Host "=== SMOKE TEST :: $dt :: $runId ==="

# 0) Compile sanity
python -m compileall .\src\engine\data .\src\engine\io .\src\engine\paradigms .\src\engine\microbatch | Out-Null
Assert-Ok "compileall"

# 1) Registry sanity (force registration before lookup)
python -c "from engine.paradigms.api import register_all_paradigms, get_hypotheses_builder; register_all_paradigms(); b=get_hypotheses_builder('ict','ict_all_windows'); print('OK builder:', getattr(b,'__name__',str(b)))"
Assert-Ok "registry sanity"

# 2) Run microbatch
python .\scripts\run_microbatch.py `
  --env $envName `
  --mode $mode `
  --snapshot-id $snapshot `
  --run-id $runId `
  --trading-day $dt `
  --cluster-id $cluster `
  --base-seed $baseSeed
Assert-Ok "run_microbatch"

# 3) Expected canonical outputs (must exist)
$paths = @(
  ".\data\features\run_id=$runId\instrument=$inst\anchor_tf=M5\dt=$dt\0000.parquet",
  ".\data\windows\run_id=$runId\instrument=$inst\anchor_tf=M5\dt=$dt\0000.parquet",
  ".\data\trade_paths\run_id=$runId\instrument=$inst\dt=$dt\0000.parquet",
  ".\data\decisions\run_id=$runId\instrument=$inst\stage=hypotheses\dt=$dt\0000.parquet",
  ".\data\decisions\run_id=$runId\instrument=$inst\stage=critic\dt=$dt\0000.parquet",
  ".\data\decisions\run_id=$runId\instrument=$inst\stage=pretrade\dt=$dt\0000.parquet",
  ".\data\decisions\run_id=$runId\instrument=$inst\stage=gatekeeper\dt=$dt\0000.parquet",
  ".\data\decisions\run_id=$runId\instrument=$inst\stage=portfolio\dt=$dt\0000.parquet"
)

Write-Host "`n=== CHECK: expected files exist ==="
foreach ($p in $paths) {
  if (!(Test-Path $p)) { throw "MISSING: $p" }
  $fi = Get-Item $p
  "{0}  {1,10} bytes  {2}" -f $fi.Name, $fi.Length, $fi.LastWriteTime
}

# 4) Hygiene: must be zero part.parquet anywhere
Write-Host "`n=== CHECK: no part.parquet anywhere ==="
$parts = Get-ChildItem .\data -Recurse -Filter "part.parquet" -ErrorAction SilentlyContinue
if ($parts -and $parts.Count -gt 0) {
  $parts | Select-Object -First 50 FullName | ForEach-Object { Write-Host $_ }
  throw "FOUND part.parquet (hygiene failed)"
}
Write-Host "OK: no part.parquet"

# 5) Read + validate parquet via validator script
Write-Host "`n=== CHECK: read parquet + key columns ==="
$validator = ".\scripts\smoke_validate_parquet.py"
if (!(Test-Path $validator)) {
  throw "Missing validator: $validator (create it first)"
}
python $validator --run-id $runId --dt $dt --inst $inst --anchor-tf "M5"
Assert-Ok "parquet read/validate"

Write-Host "`n=== SMOKE TEST PASSED ==="
