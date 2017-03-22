from  optparse import OptionParser
parser = OptionParser()
parser.add_option("-n","--sim_name", dest="l", type="str", default="[]", help='example: python parser_matrix_input.py -n [[3,2],[1,2]]')
(option,args)=parser.parse_args()
print eval(option.l)

