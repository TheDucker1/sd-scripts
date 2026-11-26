@echo off
setlocal
cd /d "%~dp0.."
set "PY=C:\Users\d\.conda\envs\ml\python.exe"
if not exist "work\danbooru-0000-0101-p20\tags-oppai-v1_1\tag-cache.json" (
  echo Finish work\danbooru-0000-0101-p20\tag.cmd first. The completed tag-cache.json is required.
  exit /b 1
)
"%PY%" tools\prepare_anima_artist_training.py
if errorlevel 1 exit /b %errorlevel%
"%PY%" -u tools\compress_anima_snapref.py ^
  --input "C:\Users\d\Documents\comfy\ComfyUI\models\loras\snapref_v2a.safetensors" ^
  --output work\anima-svd-r32-512\init\snapref-v2a-r32-r32.safetensors ^
  --ref-rank 32 --tgt-rank 32 --threads 4
if errorlevel 1 exit /b %errorlevel%
"%PY%" -u anima_train_style.py ^
  --config configs\anima_style_artist_p20_r32_200k_local.json ^
  --output_dir work\anima-artist-p20-r32-200k\run ^
  --device cuda --num_threads 4 --sdpa_backend auto ^
  --compile --compile_scope blocks --compile_modules dit ^
  --compile_backend inductor --compile_dynamic false --compile_cache_size_limit 64
exit /b %errorlevel%
