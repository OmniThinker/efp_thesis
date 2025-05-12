#!/bin/sh
randoms=(123221) # )) ) ) 1002232344665 45389 42 321234
lr=0.00005
scaling_factor=0.5
alpha=0.3
label_smoothing=0.01
test_file="en_test.json"
test=1

for random in ${randoms[@]}; do
	name="TEST_pipeline_factuality;random=${random};lr=${lr};scaling_factor=${scaling_factor};alpha=${alpha};label_smoothing=${label_smoothing}"
	sbatch pipeline.slurm --model_name='/fp/projects01/ec30/models/xlm-roberta-large/' \
                --lr=${lr} \
                --epochs=20 \
                --decay=0.01 \
                --batch_size=16 \
		            --name="${name}" \
                --random=${random} \
                --scaling_factor=${scaling_factor} \
                --alpha=${alpha} \
                --label_smoothing=${label_smoothing} \
		            --warmup_ratio=0.1 \
                --lr_strategy=reduce_lr_on_plateau \
                --test_file=${test_file} \
                --test=${test} \
                ${@}
	done
