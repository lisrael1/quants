# run by 
# python3 ../cluster_run_multi_params.py |bash

print(r'''sbatch --mem=50m -c1 --time=23:50:0 --wrap 'while [ true ] ; do squeue -u lisrael1|pee cat "grep wrap|wc -l"|mutt -s done israelilior@gmail.com;sleep 1h;done' ''')

methods='basic_method,clipping_method,sinogram_method,ml_map_method'.split(',')
snrs=[j for sub in [[5*10**i,10**i] for i in range(2,6)] for j in sub]
bins_options=[301,501]

for method in methods:
    for snr in snrs:
        for bins in bins_options:
            sim_splits=499 if method!='basic_method' else 1
            print(r'''sbatch -J {method}_{snr} --mem=1800m -c1 --time=23:50:0 --array=0-{sim_splits} --wrap 'python3 ../flow_cross_methods.py -s ${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --methods="[\"{method}\"]" -q "[0,0.25,150]" -b "[{bins}]" --snr="[{snr}]" -m 20 -n 30' '''.format(**locals()))
            
print(r'''sbatch --dependency=singleton --mem=75000m --time=23:50:0 --wrap 'python3 ../all_gz_files_to_df.py;(date;echo 'now you have pivot of the results at ~/www/pivot_rmse_*')|mutt -s done israelilior@gmail.com' ''')
# print(r'''sbatch --dependency=singleton --wrap 'date|mutt -s done israelilior@gmail.com' ''')
