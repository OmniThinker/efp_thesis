#!/bin/sh

epochs=20
batch_size=32
decay=0.01
learning_rate=0.000005 # 0.00002
model='/fp/projects01/ec30/models/xlm-roberta-large/'
concat=("type") # "trigger")
seeds=(1282) # )a 12392131 245321 4783912 123993 
export PYTHONPATH=.

for c in ${concat[@]}; do
  for random in ${seeds[@]}; do
    echo "Submitting job with parameters: model=${model}, lr=${learning_rate}, frozen=False, concat=$c, epochs=${epochs}, batch_size=${batch_size}"
    sbatch bert_baseline.slurm --model=${model} \
        --lr=${learning_rate} \
        --epochs=${epochs} \
        --name=TEST_bert_baseline \
        --decay=${decay} \
        --batch_size=${batch_size} \
        --concat=$c \
        --random=$random \
        ${@}
  done
done


