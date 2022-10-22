#!/bin/csh
#$ -M pgu@nd.edu
#$ -q gpu@qa-v100-001 -l gpu=1
#$ -m abe
#$ -r y
#$ -N REP_500_R2Net_supernova_simple_v100

module load pytorch/1.1.0	         # Required modules
python3 main.py
#tensorflow/1.3
