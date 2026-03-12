$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"

function Resolve-Python {
    $candidates = @(
        "py -3.11",
        "py -3.10",
        "python",
        "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe",
        "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
    )

    foreach ($candidate in $candidates) {
        try {
            if ($candidate -like "*.exe") {
                if (Test-Path $candidate) {
                    return $candidate
                }
                continue
            }

            $null = Invoke-Expression "$candidate --version"
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }

    throw "未找到可用的 Python 解释器。请先安装 Python 3.10/3.11。"
}

$python = Resolve-Python

if (!(Test-Path $venv)) {
    if ($python -like "*.exe") {
        & $python -m venv $venv
    } else {
        Invoke-Expression "$python -m venv `"$venv`""
    }
}

$py = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

& $py -m pip install --upgrade pip setuptools wheel
& $pip install -r (Join-Path $root "requirements.txt")

if (Test-Path (Join-Path $root "vendor\MeloTTS\setup.py")) {
    & $pip install -e (Join-Path $root "vendor\MeloTTS")
}

Write-Host "环境已准备完成。"
