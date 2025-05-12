#!/bin/sh
randoms=(42) # )) 3295992 45389 123221 932344665 932344665 1002232344665
lr=0.00004
scaling_factor=0.5
alpha=0.3
label_smoothing=0.01
test_file="en_dev.json"
test=0

for random in ${randoms[@]}; do
	name="BERT_factuality_BIO;random=${random};lr=${lr};scaling_factor=${scaling_factor};alpha=${alpha};label_smoothing=${label_smoothing};test_file=${test_file}"
	sbatch bio.slurm --model_name='/fp/projects01/ec30/models/xlm-roberta-large/' \
                --lr=0.00005 \
                --epochs=20 \
                --decay=0.01 \
                --batch_size=16 \
                --scaling_factor=${scaling_factor} \
                --alpha=${alpha} \
                --label_smoothing=${label_smoothing} \
		            --name="${name}" \
                --random=${random} \
		            --warmup_ratio=0.1 \
                --lr_strategy=reduce_lr_on_plateau \
                --test_file=${test_file} \
                --test=${test} \
                ${@}
	done
