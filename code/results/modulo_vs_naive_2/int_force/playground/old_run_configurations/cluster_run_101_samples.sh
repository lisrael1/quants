high bins and snr

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1   --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]"    -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 1  -n 10'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 10'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 40 -n 10'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]"   -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 10'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1   --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]"    -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 1  -n 20'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 20'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 40 -n 20'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]"   -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 20'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1   --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]"    -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 1  -n 30'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 30'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 40 -n 30'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]"   -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 30'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1   --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]"    -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 1  -n 40'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 40'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 40 -n 40'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]"   -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 40'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1   --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]"    -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 1  -n 100'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 100'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 40 -n 100'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 ../flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]"   -q "[0.05,0.14,15]" -b "[101]" --snr="[100000]" -m 20 -n 100'

sbatch --dependency=singleton --wrap 'python3 plot_res.py;cp pivot.csv *.html ~/www;date|mutt -s done israelilior@gmail.com'