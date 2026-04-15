@echo off
REM 一键运行默认实验矩阵（3 算法 x 2 灰尘 x 2 种子，输出到项目 batch_results）
cd /d "%~dp0\.."
python experiments\run_batch.py --preset default %*
