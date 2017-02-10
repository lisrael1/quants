#!/usr/bin/python
#run like this:
#./run_sim_2_inputs.py -h
execfile("functions/functions.py")


#next TODO:
#try to do SNR minus quantization SNR
#probably you better put a quant at the 0, but then you will always will have odd number of quants


"""
we have here only 2 inputs, y and x, and each input has its own modulo size and quantization
we want to see how the change of y modulo and quantization replects on x mse

gvim shortcuts:
zf create folding
za toggle folding state

"""



help_text='''
	run like this:

	seq 0 10   |awk '{system ("sbatch  --mem=10000m -c1 --time=0:10:0 --wrap \\"run_sim_2_inputs.py -f name_del -n sim_10k -s 5e1 -p "$1"\\"")}'
	seq 0 1259|awk '{system ("sbatch --mem=100000m -c1 --time=0:10:0 --wrap \"run_sim_2_inputs.py -f 5e6 -s 5e6 -p "$1"\"")}'
	seq 0 10027|awk '{system ("sbatch --mem=100000m -c1 --time=0:10:0 --wrap \\"run_sim_2_inputs.py -f y_quant_10k -s 5e7 -p "$1"\\"")}'

	run_sim_2_inputs.py -f try_name_del2 -n 10k -s 5e1 -p 4
	run_sim_2_inputs.py -f try_name_del2 -n 5e1 -s 5e1 

	sbatch --mem=10000m -c1 --time=0:10:0 --wrap "run_sim_2_inputs.py -f try_name_del2 -n 50 -s 5e1 -p 11"
	sbatch --mem=10000m -c1 --time=0:10:0 --wrap "run_sim_2_inputs.py -f try_name_del2 -n 1k -s 5e1"

	put at least --mem=1000m. at 100m it will fail on memory

	after running:
		 cat *csv[0-9]* |head -1 >all.csv 
		 cat *csv[0-9]* |grep -v alpha >>all.csv
		 echo ""|mutt israelilior@gmail.com -a all.csv
'''
parser = OptionParser(usage=help_text)
parser.add_option("-n","--sim_name", dest="sim_name_string", type="str", default="", help="optional. sim name at the output files")
parser.add_option("-s","--samples_per_sim", dest="samples_per_sim", type="float", help="samples per sim. 4e1 is fast, 4e6 is good, 4e7 is too much")
parser.add_option("-p","--run_only_this_itteration", dest="sim_itteration", type="int", default=-1,help="you can ignore this and it will run all cases, or you can put here number of itteration so the sim will only run this itteration. if you put -2 you will get len of options")
parser.add_option("-f","--output_folder", dest="output_folder", type="str", default="",help="creating new folder in temp for saving all results")
(option,args)=parser.parse_args()





output_folder="temp/"+option.output_folder+"/"
try:
	makedirs(output_folder)
except:
	None
#if not path.exists(output_folder):
#	print "folder "+output_folder+" doesnt exist, exit"
#	exit()



sim_name=option.sim_name_string+"_"+str(option.samples_per_sim)
output_resutls_file=output_folder+'sim_results_'+sim_name+'.csv'
output_log_file=output_folder+'sim_log_'+sim_name+'.log'

sim_log_print("momory after imports all: "+str(psutil.Process(getpid()).memory_info().rss))




max_x_bin_number=20
##max_x_bin_number=10#for fast estimation
min_x_bin_number=2
##min_x_bin_number=5#for fast estimation

number_of_samples=4e4#slow
number_of_samples=1#put here 1 and look at all_data.csv to see the flow on single sample
number_of_samples=4e6#slow, only for server
number_of_samples=4e1#fast estimation
number_of_samples=2.8e7#supper slow, always failing to me on memory size, even with 3 cpus...
number_of_samples=option.samples_per_sim#float(argv[2])

modulo_jumps_resolution=0.05
#modulo_jumps_resolution=0.5#for fast estimation
min_modulo_size=5.5
##min_modulo_size=7#from 5 quants, the modulo starts at 7

cov=mat([[10,9.5],
         [9.5,10]])

#preparing the data - try to fix this to search for the best one:
def prepare_sim_args():
    sim_log_print("simulation time start creating quantizers")
    def x_args():
        if 0:#take known best quantizer instead of looking for them
		best_bin_sizes_at_1_var={1: 0, 2: 0.0616666, 3: 1.64, 4: 1.342, 5: 1.186666, 6: 1.0021428571428579, 7: 0.91375, 8: 0.838888, 9: 0.7565, 10: 0.660454, 11: 0.6333333, 12: 0.57346153846153891, 13: 0.535, 14: 0.5106666, 15: 0.4821875, 16: 0.4529411764705884, 17: 0.434166666, 18: 0.415263157894, 19: 0.38925, 20: 0.36142857142857165, 21: 0.368181818181, 22: 0.34456521739130475, 23: 0.32666666, 24: 0.3184, 25: 0.30692307692307708, 26: 0.28555555555, 27: 0.28392857142857153, 28: 0.27741379310344838, 29: 0.265833333, 30: 0.25983870967741973, 31: 0.245625, 32: 0.2360606060606063, 33: 0.23573529411764715, 34: 0.21785714285714297, 35: 0.22208333333333341, 36: 0.21310810810810832, 37: 0.21381578947368451, 38: 0.20243589743589752, 39: 0.188}
        	qx=[simple_quantizer(number_of_quants=i,bin_size=cov[0,0]*best_bin_sizes_at_1_var[i]) for i in range(min_x_bin_number,max_x_bin_number)]
        if 1:#trying all modulo sizes (at relevant range of 4 to 9):
        	qx=[simple_quantizer(number_of_quants=j,bin_size=i/j) for i in arange(min_modulo_size,9,modulo_jumps_resolution) for j in range(min_x_bin_number,max_x_bin_number)]
        if 0:#try just a few numbers to see how the flow runs
        	qx=[quantizer(number_of_quants=5,bin_size=6.6/5) for i in range(20)] #put number_of_samples=1 and independed_var=100
        return qx

    def y_args():
    	#now lets pick y quantizer:
    	if 0:
    		qy=[simple_quantizer(number_of_quants=2000,bin_size=i) for i in [0.5,1,1.5,2,2.5,3,4,5]]
    	else:#perfect Y
    		qy=[simple_quantizer(number_of_quants=200000,bin_size=1)]
        return qy

    sim_args=[sim_2_inputs(
		number_of_samples=number_of_samples,#dont put above 4e5
		cov=cov,
		x_quantizer=i,
		y_quantizer=j,
		dither_on=0
		) for i in x_args() for j in y_args()]
    sim_log_print("simulation time finish generation sim args")
    return sim_args

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
	sim_args=[prepare_sim_args()[option.sim_itteration]]
	sim_results=run_sim(sim_args)
	output_resutls_file=output_resutls_file+str(option.sim_itteration)

sim_log_print("momory before writing results to csv: "+str(psutil.Process(getpid()).memory_info().rss))

sim_results.to_csv(output_resutls_file,mode='a')
sim_log_print("simulation end time")
