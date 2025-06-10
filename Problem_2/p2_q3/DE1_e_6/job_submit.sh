#!/bin/bash -l
#SBATCH --job-name=P2_q3_DE1
#SBATCH -t 72:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH -A sgoswam4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=droysar1@jhu.edu

source /home/droysar1/NASA_DNV/DNV_env/bin/activate

cd /home/droysar1/scr4_sgoswam4/Dibakar/DNV/Problem2_v6/p2_q3/DE1_e_6
python3 -u P2_q3_DE1_e_6.py