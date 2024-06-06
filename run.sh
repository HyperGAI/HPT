torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model hpt-edge-1-5

torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model hpt-edge-1-5

torchrun --nproc-per-node=8 run.py --data SEEDBench_IMG --model hpt-edge-1-5
