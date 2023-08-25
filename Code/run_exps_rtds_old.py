import os
#from trainer_adv import parser, train, test, get_ckp_path
from trainer import parser, train, test, get_ckp_path

args = parser.parse_args()
#args.odeMethod = 'dopri5'
args.sdeConnect = 'r'#'d'#
args.dimIn = 2#3#
args.p_spatial = 0.8#0.7#0.9#1.0#0.6#
args.p_temporal = 0.5#0.7#1.0#0.4#
#args.p = 0.7#0.9#0.1#0.3#0.5#1.0#
args.np = 0#0.01#0.03#0.1#0.05#
args.dr = 1#10#100#
args.data_path = 'datasets/RTDS'#'datasets/RTDS_Real/Processed_Data/Bus_Voltage/'#
# args.model_name = 'HGDCODE'
args.dataset = 'rtds_noisy_node'#'rtds_nodes'#'rtds_real'#'rtds_edges'#'rtds_noisy_edge'#
args.numNode = 33#32#
args.delta_t = 0.01
args.batch_size = 40#50#30#
args.numTraj = 5#10#
#args.numRamWalks = 3#1#2#
# args.contrastive = True
args.adversarial = False#True#
args.dimState = 2#3#32#
args.early_stop = 100
args.beta = 1
args.alpha = 1
args.max_epoch = 100
args.lr = 1e-2
train(args)
args.checkname = get_ckp_path(args)[0]
args.max_epoch = 100
args.lr = 1e-3
train(args)
test(args)
args.checkname = get_ckp_path(args)[0]

args.lr = 1e-4
args.delta_t = 0.001#0.01#
args.max_epoch = 100
args.batch_size = 20
train(args)
args.checkname = get_ckp_path(args)[0]
test(args)
