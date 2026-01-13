param(
  [string]$RepoRoot = (Get-Location).Path,
  [string]$OutDir   = "artifacts/code_snapshot",

  # What to include (repo-relative globs)
  [string[]]$IncludeGlobs = @(
    "src/engine/**",
    "scripts/**",
    "conf/**",
    "snapshots/**",
    "*.md",
    "*.txt"
  ),

  # Absolute path(s) to exclude completely (wins over includes)
  [string[]]$ExcludeAbsRoots = @(
    "C:\Users\ciero\Desktop\project\sandbox_data\engine_artifacts_archive"
  ),

  # Cap for each dt=... folder
  [int]$DtFolderMaxFiles = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------------
# Helpers
# -------------------------

function Normalize-FullPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $null }
  $full = (Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue)
  if ($null -eq $full) {
    # If it doesn't exist yet, normalize best-effort
    return ([System.IO.Path]::GetFullPath($p))
  }
  return $full.Path
}

function Ensure-TrailingSep([string]$p) {
  if ($p.EndsWith([System.IO.Path]::DirectorySeparatorChar)) { return $p }
  return $p + [System.IO.Path]::DirectorySeparatorChar
}

function Convert-GlobToRegex([string]$glob) {
  # Simple glob -> regex conversion supporting **, *, ?
  $r = [Regex]::Escape($glob)
  $r = $r -replace "\\\*\\\*", ".*"
  $r = $r -replace "\\\*", "[^/\\]*"
  $r = $r -replace "\\\?", "."
  return "^" + $r.Replace("\", "[\\/]").Replace("/", "[\\/]") + "$"
}

function Match-AnyGlob([string]$relNorm, [string[]]$globs) {
  foreach ($g in $globs) {
    $rx = Convert-GlobToRegex $g
    if ($relNorm -match $rx) { return $true }
  }
  return $false
}

function Is-UnderExcludedAbsRoot([string]$fullPath, [string[]]$excludedRootsNorm) {
  foreach ($root in $excludedRootsNorm) {
    if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }
  return $false
}

function Get-RepoRel([string]$repoRootNorm, [string]$fullPath) {
  $rel = $fullPath.Substring($repoRootNorm.Length).TrimStart('\','/')
  return ($rel -replace "\\","/")  # normalize to forward slashes
}

function Get-DtGroupKey([string]$relNorm) {
  # If path contains a directory segment like ".../dt=YYYY-MM-DD/..."
  # group key is everything up to and including that dt=... directory.
  $m = [Regex]::Match($relNorm, "^(.*?/dt=[^/]+)(/.*)?$", [Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if ($m.Success) { return $m.Groups[1].Value }
  return $null
}

function New-CleanDir([string]$Path) {
  if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Get-GitInfo([string]$Root) {
  $gitDir = Join-Path $Root ".git"
  if (-not (Test-Path -LiteralPath $gitDir)) {
    return @{ is_git = $false; commit = $null; branch = $null; dirty = $null }
  }
  $commit = (git -C $Root rev-parse HEAD) 2>$null | Select-Object -First 1
  $branch = (git -C $Root rev-parse --abbrev-ref HEAD) 2>$null | Select-Object -First 1
  $status = (git -C $Root status --porcelain) 2>$null
  $dirty  = (-not [string]::IsNullOrWhiteSpace(($status | Out-String)))
  return @{ is_git = $true; commit = $commit; branch = $branch; dirty = $dirty }
}

# -------------------------
# Normalize roots
# -------------------------

$RepoRoot = Normalize-FullPath $RepoRoot
$RepoRootNorm = Ensure-TrailingSep $RepoRoot

$ExcludedRootsNorm = @()
foreach ($p in $ExcludeAbsRoots) {
  $np = Normalize-FullPath $p
  if ($np) { $ExcludedRootsNorm += (Ensure-TrailingSep $np) }
}

# -------------------------
# Exclusions (junk)
# -------------------------

# Directory name exclusions (anywhere in path)
$ExcludedDirNames = @(
  "__pycache__", ".git", ".hg", ".svn",
  ".venv", "venv", "env",
  ".pytest_cache", ".mypy_cache", ".ruff_cache",
  "node_modules", "dist", "build", ".idea", ".vscode"
)

# File extension exclusions
$ExcludedExtensions = @(
  ".pyc", ".pyo", ".pyd",
  ".bak", ".tmp", ".swp", ".swo",
  ".log", ".old", ".orig"
)

# Also exclude editor/OS junk by exact file name
$ExcludedFileNames = @(
  "Thumbs.db", "Desktop.ini", ".DS_Store"
)

function Is-Junk([System.IO.FileInfo]$f, [string]$relNorm) {
  $name = $f.Name
  if ($ExcludedFileNames -contains $name) { return $true }

  $ext = [System.IO.Path]::GetExtension($name)
  if ($ExcludedExtensions -contains $ext.ToLowerInvariant()) { return $true }

  foreach ($dn in $ExcludedDirNames) {
    if ($relNorm -match "(^|/)$([Regex]::Escape($dn))(/|$)") { return $true }
  }

  return $false
}

# -------------------------
# Collect candidate files
# -------------------------

$allFiles = Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Force

$candidates = @()
foreach ($f in $allFiles) {
  $full = $f.FullName
  if (Is-UnderExcludedAbsRoot $full $ExcludedRootsNorm) { continue }

  $relNorm = Get-RepoRel $RepoRootNorm $full

  # Include filter first
  if (-not (Match-AnyGlob $relNorm $IncludeGlobs)) { continue }

  # Junk filter
  if (Is-Junk $f $relNorm) { continue }

  $candidates += [PSCustomObject]@{
    FullPath = $full
    RelNorm  = $relNorm
    LastWriteTimeUtc = $f.LastWriteTimeUtc
    Length = $f.Length
  }
}

# -------------------------
# Apply dt= folder cap (top N most recently modified per dt= dir)
# -------------------------

$dtGrouped = @{}
$nonDt = New-Object System.Collections.Generic.List[object]

foreach ($c in $candidates) {
  $k = Get-DtGroupKey $c.RelNorm
  if ($null -eq $k) {
    $nonDt.Add($c) | Out-Null
    continue
  }
  if (-not $dtGrouped.ContainsKey($k)) {
    $dtGrouped[$k] = New-Object System.Collections.Generic.List[object]
  }
  $dtGrouped[$k].Add($c) | Out-Null
}

$selected = New-Object System.Collections.Generic.List[object]
# Keep all non-dt files
foreach ($x in $nonDt) { $selected.Add($x) | Out-Null }

# For each dt= dir, take top N by LastWriteTimeUtc desc, then by path for determinism
foreach ($k in ($dtGrouped.Keys | Sort-Object)) {
  $top = $dtGrouped[$k] |
    Sort-Object @{Expression="LastWriteTimeUtc"; Descending=$true}, @{Expression="RelNorm"; Descending=$false} |
    Select-Object -First $DtFolderMaxFiles
  foreach ($x in $top) { $selected.Add($x) | Out-Null }
}

# De-dup defensively (shouldn't happen, but safe)
$selected = $selected |
  Sort-Object RelNorm -Unique

# -------------------------
# Output layout
# -------------------------

$git = Get-GitInfo $RepoRoot
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$snapshotId = if ($git.is_git -and $git.commit) { $git.commit.Substring(0,12) } else { (Get-Random).ToString() }

$OutDirAbs = Join-Path $RepoRoot $OutDir
$FilesDir  = Join-Path $OutDirAbs "FILES"

New-CleanDir $OutDirAbs
New-Item -ItemType Directory -Force -Path $FilesDir | Out-Null

# Copy selected files preserving relative paths
foreach ($s in $selected) {
  $dest = Join-Path $FilesDir ($s.RelNorm -replace "/","\")  # convert to native separators
  $destDir = Split-Path $dest -Parent
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  Copy-Item -LiteralPath $s.FullPath -Destination $dest -Force
}

# Build a snapshot-local tree from selected files (plus their parent dirs)
$treeSet = New-Object System.Collections.Generic.HashSet[string]
foreach ($s in $selected) {
  $parts = $s.RelNorm.Split("/")
  $acc = ""
  for ($i=0; $i -lt $parts.Length; $i++) {
    $acc = if ($acc -eq "") { $parts[$i] } else { "$acc/$($parts[$i])" }
    [void]$treeSet.Add($acc)
  }
}
$treePath = Join-Path $OutDirAbs "project_tree.txt"
$treeSet | Sort-Object | Set-Content -Encoding UTF8 $treePath

# Manifest with sha256 for copied files
$manifestEntries = @()
$copiedFiles = Get-ChildItem -LiteralPath $FilesDir -Recurse -File -Force
foreach ($cf in $copiedFiles) {
  $rel = $cf.FullName.Substring($FilesDir.Length).TrimStart('\','/')
  $relNorm = ("FILES/" + ($rel -replace "\\","/"))
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $cf.FullName).Hash.ToLowerInvariant()
  $manifestEntries += [PSCustomObject]@{
    path = $relNorm
    sha256 = $hash
    bytes = $cf.Length
    last_write_utc = $cf.LastWriteTimeUtc.ToString("o")
  }
}

$manifest = [PSCustomObject]@{
  snapshot_id = $snapshotId
  generated_utc = $timestamp
  repo_root = $RepoRoot
  git = [PSCustomObject]@{
    is_git = $git.is_git
    commit = $git.commit
    branch = $git.branch
    dirty  = $git.dirty
  }
  include_globs = $IncludeGlobs
  exclude_abs_roots = $ExcludedRootsNorm
  dt_folder_max_files = $DtFolderMaxFiles
  excluded_dir_names = $ExcludedDirNames
  excluded_extensions = $ExcludedExtensions
  files_count = $manifestEntries.Count
  files = ($manifestEntries | Sort-Object path)
}

$manifestPath = Join-Path $OutDirAbs "MANIFEST.json"
$manifest | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 $manifestPath

Write-Host "Wrote snapshot packet to: $OutDirAbs"
Write-Host " - project_tree.txt (snapshot-local, dt= capped)"
Write-Host " - MANIFEST.json"
Write-Host " - FILES/ (copied files: $($manifestEntries.Count))"
