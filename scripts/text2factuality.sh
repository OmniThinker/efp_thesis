#!/bin/sh

lrs=(0.00005) # 0.0001)
models=("google/flan-t5-large") #/fp/projects01/ec30/models/flan-t5-base") # "/fp/projects01/ec30/models/t5-base")

randoms=(412) # ) ) 42 9889 4231 4232
for random in ${randoms[@]}; do
	for lr in ${lrs[@]}; do
		for model in ${models[@]}; do
			sbatch text2factuality.slurm --model_name=${model} \
			--lr=${lr} \
			--lr_strategy="reduce_lr_on_plateau" \
			--epochs=15 \
			--batch_size=1 \
			--random=${random} \
			--name="test-text2factuality-exp" \
			${@}
		done
	done
done

