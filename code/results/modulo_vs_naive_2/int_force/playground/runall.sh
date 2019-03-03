# run me with nohup bash runall.sh&
sleep 70000
python3 plot_res.py 
cp pivot.csv temp-plot.html ~/www
date|mutt -s done israelilior@gmail.com
