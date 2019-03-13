# run me with nohup bash runall.sh&
# but now you have it at the cluster command, using dependency...
sleep 10000
python3 plot_res.py 
cp pivot.csv *.html ~/www
date|mutt -s done israelilior@gmail.com
sleep 10000
python3 plot_res.py 
cp pivot.csv *.html ~/www
date|mutt -s done israelilior@gmail.com
sleep 10000
python3 plot_res.py 
cp pivot.csv *.html ~/www
date|mutt -s done israelilior@gmail.com
