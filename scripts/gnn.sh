#!/bin/sh
 
# batch_size     Batch size
# learning_rate  Learning rate
epochs=100
batch_size=16
seeds=(4783912 12392131) #)  )245321 42 1282 123993
lr=0.00004
lr_bert=0.000005
lr_patience=10
stopping_patience=15
stopping_min_delta=5
dropout=0.5
hidden_dim=128
model='/fp/projects01/ec30/models/xlm-roberta-large/'


export PYTHONPATH=.
for random in ${seeds[@]}; do
  name="TEST_GNN_attention_with_lr;random=${random};model=${model};lr=${lr};lr_bert=${lr_bert};hidden_dim=${hidden_dim};epochs=${epochs};lr_patience=${lr_patience};stopping_patience=${stopping_patience};stopping_min_delta=${stopping_min_delta};dropout=${dropout}"
  echo "Submitting job with parameters: ${name}"
  sbatch train_gnn.slurm --epochs=${epochs} \
      --batch_size=${batch_size} \
      --random=${random} \
      --lr=${lr} \
      --lr_bert=${lr_bert} \
      --lr_patience=${lr_patience} \
      --stopping_patience=${stopping_patience} \
      --stopping_min_delta=${stopping_min_delta} \
      --dropout=${dropout} \
      --hidden_dim=${hidden_dim} \
      --model=${model} \
      --name="${name}" \
      ${@}
done


