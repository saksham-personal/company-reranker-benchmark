param(
    [Parameter(Mandatory=$true)][string]$WorkDir,
    [string]$ArchivePath,
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'configs\assets.json') -Raw | ConvertFrom-Json
$asset = @($manifest.assets | Where-Object { $_.id -eq 'offline-python-windows-x64' })
if ($asset.Count -ne 1) { throw 'The pinned offline runtime asset is missing.' }
$asset = $asset[0]
$workRoot = [System.IO.Path]::GetFullPath($WorkDir)
New-Item -ItemType Directory -Force -Path $workRoot | Out-Null
$archive = if ($ArchivePath) { [System.IO.Path]::GetFullPath($ArchivePath) } else { Join-Path $workRoot $asset.asset }
if (-not (Test-Path -LiteralPath $archive)) {
    $url = "https://github.com/$($asset.github.repository)/releases/download/$($asset.github.release)/$($asset.asset)"
    Invoke-WebRequest -Uri $url -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.archive_sha256) {
    throw 'Offline runtime archive checksum mismatch.'
}
$bundle = Join-Path $workRoot 'offline-environment'
if (-not (Test-Path -LiteralPath $bundle)) { Expand-Archive -LiteralPath $archive -DestinationPath $bundle }
$bundleFull = [System.IO.Path]::GetFullPath($bundle).TrimEnd('\')
foreach ($record in $asset.files) {
    $file = [System.IO.Path]::GetFullPath((Join-Path $bundle $record.path))
    if (-not $file.StartsWith($bundleFull + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Payload path escapes bundle.' }
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing offline payload: $($record.path)" }
    if ((Get-Item -LiteralPath $file).Length -ne $record.size_bytes -or
        (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) {
        throw "Offline payload checksum mismatch: $($record.path)"
    }
}
$python = Join-Path $bundle 'python\python.exe'
$venv = Join-Path $workRoot '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not $SkipInstall) {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $python -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    & $venvPython -m pip install --no-index --find-links (Join-Path $bundle 'wheels') --require-hashes -r (Join-Path $bundle 'requirements.lock')
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency install failed.' }
}
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HOME = Join-Path $workRoot 'cache\huggingface'
$env:HF_MODULES_CACHE = Join-Path $workRoot 'cache\modules'
Write-Output "Ready. Python: $venvPython"
