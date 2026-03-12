@echo off
setlocal

set "PYTHON_EXE="

if exist "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe" set "PYTHON_EXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" set "PYTHON_EXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"

if not defined PYTHON_EXE (
  echo 未找到 Python 3.10 或 3.11，请先安装 Python。
  exit /b 1
)

"%PYTHON_EXE%" -m venv ".venv"
if errorlevel 1 (
  echo 创建虚拟环境失败。
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo 升级 pip 失败。
  exit /b 1
)

".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
  echo 安装 requirements.txt 失败。
  exit /b 1
)

if exist "vendor\MeloTTS\setup.py" (
  ".venv\Scripts\pip.exe" install -e "vendor\MeloTTS"
  if errorlevel 1 (
    echo 安装 MeloTTS 失败。
    exit /b 1
  )
)

echo 环境已准备完成。
endlocal
