@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo 未找到 .venv，请先运行 scripts\setup_portable.cmd
  exit /b 1
)

".venv\Scripts\python.exe" -m app
endlocal
