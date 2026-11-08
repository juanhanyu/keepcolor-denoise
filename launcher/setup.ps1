# Denoise first-time setup (Denoise.exe opens this in a console window).
# 1. uv, the Python installer: from PyPI, the Tsinghua mirror or GitHub, whichever answers.
# 2. A private Python 3.12 in runtime\python (GitHub, then a mirror; the system's Python 3.12 as a last resort).
# 3. setup_env.py: picks the PyTorch build for the graphics card, installs everything, downloads models, opens Denoise.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is far faster without its progress bar
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = 'Denoise 首次安装'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$app = Split-Path -Parent $PSScriptRoot
$rt = Join-Path $app 'runtime'
New-Item -ItemType Directory -Force -Path $rt | Out-Null

function Fail($msg) {
  Write-Host ''
  Write-Host $msg -ForegroundColor Red
  Read-Host '按回车关闭'
  exit 1
}

$uv = Join-Path $rt 'uv\uv.exe'
if (-not (Test-Path $uv)) {
  Write-Host '下载 uv（Python 安装工具，约 20 MB）...' -ForegroundColor Cyan
  $sources = @()
  try {
    $sources += ((Invoke-RestMethod 'https://pypi.org/pypi/uv/json' -TimeoutSec 30).urls |
      Where-Object { $_.filename -like '*-win_amd64.whl' } | Select-Object -First 1).url
  } catch {}
  try {
    $index = 'https://pypi.tuna.tsinghua.edu.cn/simple/uv/'
    $links = [regex]::Matches((Invoke-WebRequest $index -UseBasicParsing -TimeoutSec 30).Content, 'href="([^"#]+-win_amd64\.whl)')
    if ($links.Count) { $sources += [Uri]::new([Uri]$index, $links[$links.Count - 1].Groups[1].Value).AbsoluteUri }
  } catch {}
  $sources += 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
  $zip = Join-Path $rt 'uv-download.zip'
  $unpack = Join-Path $rt 'uv-unpack'
  foreach ($src in $sources) {
    if (-not $src) { continue }
    try {
      Write-Host "  从 $(([Uri]$src).Host) 下载"
      Invoke-WebRequest $src -OutFile $zip -UseBasicParsing -TimeoutSec 600
      if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
      Expand-Archive $zip -DestinationPath $unpack -Force  # a wheel is a zip file
      $exe = Get-ChildItem $unpack -Recurse -Filter 'uv.exe' | Select-Object -First 1
      if ($exe) {
        New-Item -ItemType Directory -Force -Path (Split-Path $uv) | Out-Null
        Move-Item $exe.FullName $uv -Force
        break
      }
    } catch { Write-Host "  失败：$($_.Exception.Message)" -ForegroundColor Yellow }
  }
  Remove-Item $zip, $unpack -Recurse -Force -ErrorAction SilentlyContinue
  if (-not (Test-Path $uv)) { Fail 'uv 下载失败。请检查网络后重新打开 Denoise.exe。' }
}

$env:UV_CACHE_DIR = Join-Path $rt 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $rt 'python'
$env:UV_NO_CONFIG = '1'
$env:DENOISE_UV = $uv
$env:PYTHONUTF8 = '1'
$ErrorActionPreference = 'Continue'  # from here on the tools report their own errors

Write-Host '准备 Python 3.12 ...' -ForegroundColor Cyan
$pref = 'system'
foreach ($mirror in @('', 'https://registry.npmmirror.com/-/binary/python-build-standalone')) {
  if ($mirror) { $env:UV_PYTHON_INSTALL_MIRROR = $mirror; Write-Host '  换国内镜像再试 ...' -ForegroundColor Yellow }
  & $uv python install 3.12
  if ($LASTEXITCODE -eq 0) { $pref = 'only-managed'; break }
}
if ($pref -eq 'system') { Write-Host '  下载不了独立的 Python，改用电脑上已装的 Python 3.12。' -ForegroundColor Yellow }

& $uv run --no-project --python 3.12 --python-preference $pref (Join-Path $app 'setup_env.py')
if ($LASTEXITCODE -ne 0) { Fail '安装没有完成，见上面的信息。网络不稳时重新打开 Denoise.exe 会接着装。' }
Write-Host ''
Write-Host '这个窗口 6 秒后自动关闭。' -ForegroundColor Green
Start-Sleep 6
