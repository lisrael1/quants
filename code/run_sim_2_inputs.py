#!/usr/bin/python
#run like this:
#./run_sim_2_inputs.py -h
execfile("functions/functions.py")


#next TODO:
#try to do SNR minus quantization SNR
#probably you better put a quant at the 0, but then you will always will have odd number of quants


"""
gvim shortcuts:
zf create folding
za toggle folding state

"""



help_text='''
	
	we have here only 2 inputs, y and x, and each input has its own modulo size and quantization
	we want to see how the change of y modulo and quantization replects on x mse

	run like this:
		run_sim_2_inputs.py -f my_output_folder -s 5e1 -p 4
		run_sim_2_inputs.py -f try_name_del2 -n 10k -s 5e1 -p 4
		run_sim_2_inputs.py -f try_name_del2 -n 5e1 -s 5e1 
		run_sim_2_inputs.py -f my_output_folder -s 5e4 -c [[10.1,9.5],[9.5,10.2]] -x [5000,0.01] -y [10000,0.01]

	for using cluster:
		entering cluster:
			ssh -XC sed-gw -> then enter your linux password
			hostname to see that you're in...
		seq 0 10  |awk '{system ("sbatch  --mem=10000m -c1 --time=0:10:0 --wrap \\"run_sim_2_inputs.py -f name_del -n sim_10k -s 5e1 -p "$1"\\"")}'
		seq 0 1259|awk '{system ("sbatch --mem=100000m -c1 --time=0:10:0 --wrap \\"run_sim_2_inputs.py -s 5e6 -p "$1"\\"")}'
		seq 0 1259|awk '{system ("sbatch --mem=100000m -c1 --time=0:10:0 --wrap \\"run_sim_2_inputs.py -f y_quant_10k -s 5e7 -p "$1"\\"")}'
	
	
	sbatch --mem=10000m -c1 --time=0:10:0 --wrap "run_sim_2_inputs.py -f try_name_del2 -n 50 -s 5e1 -p 11"
	sbatch --mem=10000m -c1 --time=0:10:0 --wrap "run_sim_2_inputs.py -f try_name_del2 -n 1k -s 5e1"

	put at least --mem=1100m. at 100m it will fail on memory
		up to 5e5 use 1100
		for 5e6 use 10G
		for 5e7 use 100G, try also 20G


	after running the cluser:
		cd ./temp/<your output folder>
		cat *csv[0-9]* |head -1 >all.csv
		cat *csv[0-9]* |grep -v alpha >>all.csv
		mkdir del
		mv *csv[0-9]* del
		(or in 1 line:) cat *csv[0-9]* |head -1 >all.csv && cat *csv[0-9]* |grep -v alpha >>all.csv && mkdir del && mv *csv[0-9]* del
		(or ) cat *csv[0-9]* |head -1 >all.csv && cat *csv[0-9]* |grep -v alpha >>all.csv && rm *csv[0-9]*
		echo ""|mutt israelilior@gmail.com -s 5e6 -a all.csv
		then run parse_sim_2_inputs.py manually 
'''
parser = OptionParser(usage=help_text)
parser.add_option("-n","--sim_name", dest="sim_name_string", type="str", default="", help="optional. sim name at the output files")
parser.add_option("-s","--samples_per_sim", dest="samples_per_sim", type="float", help="samples per sim. 4e1 is fast, 4e6 is good, 4e7 is too much. if you put less than 20 you will see the randomed numbers and all the flow at the csv")
'''when running on local pc:
	number_of_samples=4e1#fast estimation
	number_of_samples=4e4#slow
	number_of_samples=1#put here 1 and look at all_data.csv to see the flow on single sample
	number_of_samples=4e6#slow, only for server. for pc dont put above 4e5
	number_of_samples=2.8e7#supper slow, always failing to me on memory size, even with 3 cpus...
'''
parser.add_option("-p","--run_only_this_itteration", dest="sim_itteration", type="int", default=-1,help="the simulation will split your quantizations for multi simulations, by bin size and number. for example, 3 bins at size of 0.5 will be 1 run and 3 bins with 0.44 size will be another one. if y also have quantizer it will multiply the sim number. you can ignore this and it will run all cases without splitting, or you can put here number of itteration so the sim will only run this itteration (for cluster use...). if you put -2 you will only print you the number of split sim, if output is 306 you can run from 0 to 305")
parser.add_option("-f","--output_folder", dest="output_folder", type="str", default="last_run",help="creating new folder in ./ for saving all results. at full path it will generate the full path, even sub sub folders... dont put it at /tmp because each machine has it own /tmp")
parser.add_option("-c","--cov_matrix", dest="cov_matrix", type="str", default="-1", help="example: -c [[3,2],[1,2]]")
parser.add_option("-x","--x_bins_num_and_size", dest="qx", type="str", default="-1", help="example for 3 bins at bin size of 0.5: -x [3,0.5]")
parser.add_option("-y","--y_bins_num_and_size", dest="qy", type="str", default="-1", help="example for 3 bins at bin size of 0.5: -y [3,0.5]")
parser.add_option("-v","--vectors_max_length_to_save", dest="vectors_max_length_to_save", type="float", default=20,help="you can put here a number bigger than the number of sampling and you will see all the samples at all stages at the csv output file")

(option,args)=parser.parse_args()





output_folder=option.output_folder+"_"+str("{:.2e}".format(option.samples_per_sim))+"/"
try:
	makedirs(output_folder)
except:
	None



sim_name=option.sim_name_string+"_"+str(option.samples_per_sim)
output_resutls_file=output_folder+'sim_results_'+sim_name+'.csv'
output_log_file    =output_folder+'sim_log_'+sim_name+'.log'

sim_log_print("memory after imports all: "+str(psutil.Process(getpid()).memory_info().rss))




max_x_bin_number=20
##max_x_bin_number=10#for fast estimation
min_x_bin_number=2
##min_x_bin_number=5#for fast estimation


modulo_jumps_resolution=0.05
#modulo_jumps_resolution=0.5#for fast estimation 
min_modulo_size=5.5
#min_modulo_size=10.5
##min_modulo_size=7#from 5 quants, the modulo starts at 7
max_modulo_size=9.0
#max_modulo_size=100.0

cov=mat([[10,9.5],
         [9.5,10]])

#preparing the data - try to fix this to search for the best one:
def prepare_sim_args():
	sim_log_print("simulation time start creating quantizers")
	def set_args_from_here():
	    global qx,qy
	    def x_args():
		if 0:#take known best quantizer instead of looking for them. this is only when y quantizer is perfect...
			best_bin_sizes_at_1_var={1: 0, 2: 0.0616666, 3: 1.64, 4: 1.342, 5: 1.186666, 6: 1.0021428571428579, 7: 0.91375, 8: 0.838888, 9: 0.7565, 10: 0.660454, 11: 0.6333333, 12: 0.57346153846153891, 13: 0.535, 14: 0.5106666, 15: 0.4821875, 16: 0.4529411764705884, 17: 0.434166666, 18: 0.415263157894, 19: 0.38925, 20: 0.36142857142857165, 21: 0.368181818181, 22: 0.34456521739130475, 23: 0.32666666, 24: 0.3184, 25: 0.30692307692307708, 26: 0.28555555555, 27: 0.28392857142857153, 28: 0.27741379310344838, 29: 0.265833333, 30: 0.25983870967741973, 31: 0.245625, 32: 0.2360606060606063, 33: 0.23573529411764715, 34: 0.21785714285714297, 35: 0.22208333333333341, 36: 0.21310810810810832, 37: 0.21381578947368451, 38: 0.20243589743589752, 39: 0.188}
			qx=[simple_quantizer(number_of_quants=i,bin_size=cov[0,0]*best_bin_sizes_at_1_var[i]) for i in range(min_x_bin_number,max_x_bin_number)]
		if 1:#trying all modulo sizes (at relevant range of min_modulo_size to max_modulo_size):
			qx=[simple_quantizer(number_of_quants=j,bin_size=i/j) for i in arange(min_modulo_size,max_modulo_size,modulo_jumps_resolution) for j in range(min_x_bin_number,max_x_bin_number)]
		if 0:#try just a few numbers to see how the flow runs
			qx=[quantizer(number_of_quants=5,bin_size=6.6/5) for i in range(20)] 
		return qx

	    def y_args():
		#now lets pick y quantizer:
		if 0:
			qy=[simple_quantizer(number_of_quants=2000,bin_size=i) for i in [0.5,1,1.5,2,2.5,3,4,5]]
		else:#perfect Y
			qy=[simple_quantizer(number_of_quants=200000,bin_size=0)]
		return qy
	    qx=x_args()
	    qy=y_args()
	def set_args_from_outside_parser():
	    	global qx,qy,cov
		def read_quantizer_from_parser(q):
			try:
				q=eval(q)
				return q[0],q[1]
			except:
				print "input should be a list like this: [4,0.1]. input is ",q
				exit()
		number_of_quants,bin_size=read_quantizer_from_parser(option.qx)			
		qx=[simple_quantizer(number_of_quants=number_of_quants,bin_size=bin_size)]
		number_of_quants,bin_size=read_quantizer_from_parser(option.qy)			
		qy=[simple_quantizer(number_of_quants=number_of_quants,bin_size=bin_size)]
		cov=mat(eval(option.cov_matrix))
	if (option.qx!="-1" and option.qy!="-1" and option.cov_matrix!="-1"):
		set_args_from_outside_parser()
	else:
		set_args_from_here()
        sim_args=[sim_2_inputs(
            	number_of_samples=option.samples_per_sim,
            	cov=cov,
            	x_quantizer=i,
            	y_quantizer=j,
            	dither_on=0,
		vectors_max_length_to_save=option.vectors_max_length_to_save
            	) for i in qx for j in qy]
        return sim_args
	sim_log_print("simulation time finish generation sim args")

def run_sim(sim_args):
    list_of_df=matching(run_single_sim,sim_args)
    sim_results=pd.concat(list_of_df,axis=0)
    return sim_results

#run sim
if option.sim_itteration==-1:
	sim_results=run_sim(prepare_sim_args())
elif option.sim_itteration==-2:
	print len(prepare_sim_args())
	exit()
else:
	prepared_sim_args=prepare_sim_args()
	if len(prepared_sim_args)<=option.sim_itteration:
		print "at your sim you have only ",len(prepared_sim_args)," itterations, you cannot choose itteration number ",option.sim_itteration
		exit()
	sim_args=[prepared_sim_args[option.sim_itteration]]
	sim_results=run_sim(sim_args)
	output_resutls_file=output_resutls_file+str(option.sim_itteration)

sim_log_print("memory before writing results to csv: "+str(psutil.Process(getpid()).memory_info().rss))
#we cannot transpose because some simulations are running splitted 
#if sim_results.index.size<5:
#		sim_results=sim_results.transpose()
sim_results.to_csv(output_resutls_file,mode='a')
sim_log_print("simulation end time")
