@echo off
rem Double-click to start Dreamer (or use the "Dreamer" shortcut on the desktop).
rem Close this window, or press Ctrl+C in it, to stop Dreamer.
title Dreamer
cd /d "%~dp0"
powershell.exe -NoProfile -File "%~dp0run.ps1"
if errorlevel 1 pause
