#!/bin/bash -l
#SBATCH --job-name=bound_final
#SBATCH -t 48:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH -A sgoswam4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=droysar1@jhu.edu

source /home/droysar1/NASA_DNV/DNV_env/bin/activate

cd /home/droysar1/scr4_sgoswam4/Dibakar/DNV/Problem1_c_final
python3 -u Bound_Q1_c_res.py

