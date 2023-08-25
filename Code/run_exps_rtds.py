import os
#from trainer_adv import parser, train, test, get_ckp_path
from trainer import parser, train, test, get_ckp_path

args = parser.parse_args()
#args.odeMethod = 'dopri5'
args.sdeConnect = 'd'#'r'#
args.dimIn = 3#2#
#args.p_spatial = 0.9#1.0#0.6#0.8#0.7#
#args.p_temporal = 0.7#1.0#0.4#0.5#
args.p = 0.1#0.3#0.5#0.7#0.9#1.0#
args.np = 0#0.01#0.03#0.1#0.05#
args.dr = 10#100#
args.data_path = 'datasets/RTDS_Real/Processed_Data/Bus_Voltage/'#'datasets/RTDS'#
# args.model_name = 'HGDCODE'
args.dataset = 'rtds_real'#'rtds_edges'#'rtds_nodes'#
args.delta_t = 2.5e-3 * args.dr#0.01#0.005#0.001# 
args.batch_size = 60#30#
args.numTraj = 5
args.numRamWalks = 3#1#2#
# args.contrastive = True
# args.adversarial = True
args.dimState = 3#2#

args.beta = 1.0

args.lr = 1e-2
train(args)
args.checkname = get_ckp_path(args)[0]
args.lr = 1e-3
train(args)
args.checkname = get_ckp_path(args)[0]
args.lr = 1e-4
args.delta_t = 2.5e-4 * args.dr#0.001#
args.batch_size = 40
train(args)
test(args)