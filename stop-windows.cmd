@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-windows.ps1" %*
pause
