#!/bin/bash
i=$1
max=`run_sim_2_inputs.py -i configs/$i -p -2`
seq 0 $((max-1))| xargs -I^ sbatch --mem=10000m -c1 --time=0:10:0 --wrap "run_sim_2_inputs.py -i configs/$i -p ^"

