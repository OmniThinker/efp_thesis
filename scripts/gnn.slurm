#!/bin/bash
#SBATCH --job-name=run_gnn
#SBATCH --account=ec30
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ifi_accel #ifi_accel # accel    # ifi_accel To use the accelerator nodes
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --chdir=/fp/homes01/u01/ec-pafastin/certainty/scripts
#SBATCH -o ../slurm_outs/%j-%x.out
#SBATCH --mail-user=peder.august@protonmail.com
#SBATCH --mail-type=ALL
# NB: this script should be run with "sbatch sample.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-datasets/2.15.0-foss-2022b-Python-3.10.8
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/03-foss-2022b-Python-3.10.8
module load nlpl-wandb/0.15.2-foss-2022b-Python-3.10.8
module load nlpl-torch-geometric/2.5.3-foss-2022b-Python-3.10.8


# print information (optional)

export PYTHONPATH="${PYTHONPATH}:."
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "submission directory: ${SUBMITDIR}"

python3 ./train_gnn.py \
        --name=gnn_simple \
        ${@}

