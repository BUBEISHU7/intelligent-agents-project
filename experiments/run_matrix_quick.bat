@echo off
REM 一键快速冒烟矩阵（2 算法 x 1 规模 x 1 种子）
cd /d "%~dp0\.."
python experiments\run_batch.py --preset quick %*
