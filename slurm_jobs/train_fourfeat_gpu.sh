#!/usr/bin/sh

#SBATCH -p gpu20
#SBATCH -t 02:00:00
#SBATCH -J OctaveSimplexPerlin
#SBATCH -D /HPS/deep_topopt/work/procedural-noises-fourfeat
#SBATCH -o /HPS/deep_topopt/work/procedural-noises-fourfeat/logs/slurm/fourfeat/slurm-%x-%j.log
#SBATCH --gres gpu:1

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate pnf_py37


# call your program here
python3 /HPS/deep_topopt/work/procedural-noises-fourfeat/training/train_fourfeat.py --jid ${SLURM_JOBID}
