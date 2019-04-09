high bins and snr
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]" -q "[0,0.025,150]" -b "[1024,5001]" --snr="[10000]" -m 1'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0,0.025,150]" -b "[1024,5001]" --snr="[10000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0,0.025,150]" -b "[1024,5001]" --snr="[1000001,10000000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]" -q "[0,0.025,150]" -b "[1024,5001]" --snr="[1000001,10000000]" -m 2'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]" -q "[0,0.07,150]" -b "[511]" --snr="[10000]" -m 1'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0,0.07,150]" -b "[511]" --snr="[10000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0,0.07,150]" -b "[511]" --snr="[1000001,10000000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]" -q "[0,0.07,150]" -b "[511]" --snr="[1000001,10000000]" -m 2'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]" -q "[0,0.5,150]" -b "[101]" --snr="[100000]" -m 1'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0,0.5,150]" -b "[101]" --snr="[100000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0,0.5,150]" -b "[101]" --snr="[100000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]" -q "[0,0.5,150]" -b "[101]" --snr="[100000]" -m 2'

sbatch --mem=1800m -c1 --time=23:50:0 --array=0-1 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"basic_method\"]" -q "[0,0.5,150]" -b "[21]" --snr="[100000]" -m 1'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"clipping_method\"]" -q "[0,0.5,150]" -b "[21]" --snr="[100000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"sinogram_method\"]" -q "[0,0.5,150]" -b "[21]" --snr="[100000]" -m 2'
sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 flow_cross_methods.py -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --methods="[\"ml_map_method\"]" -q "[0,0.5,150]" -b "[21]" --snr="[100000]" -m 2'

sbatch --dependency=singleton --wrap 'python3 plot_res.py;cp pivot.csv *.html ~/www;date|mutt -s done israelilior@gmail.com'
