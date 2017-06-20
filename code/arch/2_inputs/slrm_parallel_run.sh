#!/bin/bash
#run like this:
#slrm_parallel_run.sh configs/wide_modulo_for_hist.ini |less
#and then hit shift f

i=$1

#this random number is for checking status of running sims
r=li`head -c 10000 /dev/urandom | tr -dc _A-Za-z0-9 | head -c5`

max=`run_sim_2_inputs.py -i $i -p -2`
echo "running $max simulations:"
seq 0 $((max-1))| xargs -I^ sbatch --mem=10000m -c1 --time=0:10:0 -J $r --wrap "run_sim_2_inputs.py -i $i -p ^"

echo "waiting for $max simulations to finish"
a=1
while [ $a -gt 0 ] 
	do 
	a=`squeue -u lisrael1|grep $r|wc -l`
	echo "$a sim lefted"
	sleep 1
	done
echo "removing cluster logs:"
find . -maxdepth 1 -name "slurm-5*" -empty -delete

echo "grouping all output csvs into all.csv and sending it to current folder:"
f=`cat $i|grep output_folder|cut -d = -f 2`
f=$f/`ls -t1 --group-directories-first $f|sed -n 1p`
echo "results directory: $f"
cd $f
cat *csv[0-9]* |head -1 >all.csv && cat *csv[0-9]* |grep -v alpha >>all.csv && rm *csv[0-9]*
cd -
cp $f/all.csv .

echo "done"
