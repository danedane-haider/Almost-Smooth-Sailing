#!/bin/bash

#SBATCH -p g                    # to get on part g
#SBATCH --gres=gpu:1
 #SBATCH --constraint=g3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --qos=short

~/.conda/envs/randomfilterbanks/bin/python main.py --beta 0.1