# run by 
# ssh phoenix-gw
# python3 ../cluster_run_multi_params.py |bash
import os
script_path=os.path.realpath(__file__).replace('\\', '/').rsplit('cluster_run_multi_params.py', 1)[0]


print(r'''sbatch --mem=50m -c1 --time=23:50:0 --wrap 'sleep 1h;while [ `squeue -u $USER |grep -q $USER && echo 1` ] ; do squeue -u lisrael1|pee cat "grep wrap|wc -l"|mutt -s sbatch_status israelilior@gmail.com;sleep 1h;done' ''')

methods='basic_method,clipping_method,sinogram_method,ml_map_method'.split(',')
snrs=[250,750,2500]
bins_options=[101,301,501]

# methods=['sinogram_method']*20
# snrs=[int(1e6)]
# bins_options=[301]

for method in methods:
    for snr in snrs:
        for bins in bins_options:
            sim_splits=499 if method!='basic_method' else 1
            print(r'''sbatch -J {method}_{snr} --mem=1800m -c1 --time=23:50:0 --array=0-{sim_splits} --wrap 'python3 {script_path}/flow_cross_methods.py -s ${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --methods="[\"{method}\"]" -q "[0,0.25,150]" -b "[{bins}]" --snr="[{snr}]" -m 20 -n 30' '''.format(**locals()))

print()     
print(r'''sbatch --dependency=singleton --mem=75000m --time=23:50:0 --wrap 'python3 {script_path}/all_gz_files_to_df.py;date|mutt -s done israelilior@gmail.com -a ~/www/pivot_rmse_last_one.csv' '''.format(**locals()))

print('\nfor running time:')
print('''awk '/closing/{v=$(NF-1);if (v>max)max=v;s+=v;f+=1}END{print "max "max/60"[m], average "s/f/60"[m], "f" files, total "s/60/60/24" days"}' slurm*out''')
print('\tyou will get something like:')
print('''\tmax 15.9261[m], average 1.63323[m], 5397 files, total 6.1212 days''')