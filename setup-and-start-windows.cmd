@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-and-start-windows.ps1" %*
pause
