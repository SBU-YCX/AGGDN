import os
from trainer import parser, train, test, get_ckp_path

args = parser.parse_args()
args.dimIn = 2
args.p_spatial = 0.8
args.p_temporal = 0.5
args.np = 0
args.dr = 1
args.data_path = 'datasets/RTDS'
args.dataset = 'rtds_noisy_node'
args.numNode = 33
args.delta_t = 0.01
args.batch_size = 40
args.numTraj = 5
args.adversarial = True
args.dimState = 2
args.early_stop = 20
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
args.delta_t = 0.001
args.max_epoch = 100
args.batch_size = 20
train(args)
args.checkname = get_ckp_path(args)[0]
test(args)
