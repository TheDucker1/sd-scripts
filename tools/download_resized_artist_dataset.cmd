@echo off
setlocal
cd /d "%~dp0.."
if not defined HF_XET_HIGH_PERFORMANCE set "HF_XET_HIGH_PERFORMANCE=1"
"C:\Users\d\.conda\envs\ml\python.exe" -u tools\download_resized_artist_dataset.py %*
exit /b %errorlevel%
