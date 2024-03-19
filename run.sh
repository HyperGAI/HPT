torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model hpt-air-mmmu

torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model hpt-air-mmbench

torchrun --nproc-per-node=8 run.py --data SEEDBench_IMG --model hpt-air-seed
