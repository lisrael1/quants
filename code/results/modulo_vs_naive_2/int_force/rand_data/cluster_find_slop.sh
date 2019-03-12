sbatch --mem=1800m -c1 --time=23:50:0 --array=0-499 --wrap 'python3 find_slop.py'
