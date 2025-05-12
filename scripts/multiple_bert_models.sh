#!/bin/sh
 
# batch_size     Batch size
# learning_rate  Learning rate
epochs=50
batch_size=32
decay=0.01
learning_rates=(0.000005 0.00002 0.00006)
models=('/fp/projects01/ec30/models/xlm-roberta-large/')
frozen=(False)
concat=('type')
export PYTHONPATH=.

for c in ${concat[@]}; do
  for f in ${frozen[@]}; do
      for learning_rate in ${learning_rates[@]}; do
        for model in ${models[@]}; do
          if [ $f == 'True' ]; then
            echo "Submitting job with parameters: model=${model}, lr=${learning_rate}, frozen=True, concat=$c, epochs=${epochs}, batch_size=${batch_size}"
            sbatch bert_baseline.slurm --model=${model} \
                --lr=${learning_rate} \
                --frozen=True \
                --epochs=${epochs} \
                --decay=${decay} \
                --batch_size=${batch_size} \
                --concat=$c \
                ${@}
            else
            echo "Submitting job with parameters: model=${model}, lr=${learning_rate}, frozen=False, concat=$c, epochs=${epochs}, batch_size=${batch_size}"
            sbatch bert_baseline.slurm --model=${model} \
                --lr=${learning_rate} \
                --epochs=${epochs} \
                --decay=${decay} \
                --batch_size=${batch_size} \
                --concat=$c \
                ${@}
          fi
        done
    done
  done
done


