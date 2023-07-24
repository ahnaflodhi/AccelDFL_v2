import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# # #The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys, argparse
import pickle, resource
import time
import gc, warnings
from get_args import arg_parser
from utils import save_file, model_size, combine_results
from data_utils import * # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from env_sysmodel import system_model, FL_Modes
from devices import Nodes, Servers
import torchvision.models as models
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

args = arg_parser()
dataset = args.d
batch_size = args.b
nodes = args.n
clusters = args.c
epochs_min = args.emin
epochs_max = args.emax
rounds = args.r
overlap_factor = args.o
skew =args.s
test_batch_size = args.t
prop = args.prop
agg_prop = args.aggprop
servers = args.ser
modeltype = args.model
alpha = args.a
stale = args.D
cos_lim = args.sim
prob_int = args.clint
prob_ext = args.clext

#  'ch_d2d':None, 'int_ch': None, 'gossip':None, 'hgossip':None, 'd2d':None, 'chd2d':None, 'intch': None, 'intchd2d': None, 'gossip' : None,  'cfl': None, 'sgd' : None
# modes_list = {'chd2d':None, 'intch': None, 'intchd2d': None, 'd2d':None, 'cfl': None, 'sgd' : None}
# 'd2d': d2d_flags, 'hd2d': hd2d_flags, 'hfl': hfl_flags, 'chd2d':chd2d_flags, 'intch': intch_flags, 'intchd2d':intchd2d_flags, 'gossip':gossip_flags, 'hgossip':hgossip_flags, 'cfl':cfl_flags, 'sgd':None

# Aggregation Schemes
# modes_list = {'d2dsel':None, 'd2dlay':None, 'staled2d':None, 'd2d':None, 'hfl':None, 'cfl':None}
# modes_list = {'intchd2d':None, 'intchd2dsel':None, 'd2dlay':None, 'd2d':None,}
# modes_list  = {'intchd2d':None, 'd2d':None, 'intchgossip':None}

# Node Selection Modes
# modes_list = {'d2dstalemodel':None, 'd2dsel':None, 'd2dacc':None,  'd2d':None, 'gossip':None}
modes_list = {'d2daccwtexp':None, 'd2daccwtnorm':None, 'd2daccwtskew':None, 'd2daccwtlin':None, 'd2dacc':None, 'd2dloss':None, 'd2d':None}


# Memory and Relay Modes
# modes_list = {'d2dexnhbr':None, 'd2dmgs':None, 'd2d':None}
# modes_list = {'d2dfullmem':None,'d2dexnhbr':None, 'd2dmemextndrly':None, 'd2dmem':None, 'd2d':None} #   'd2dexrly':None, 'd2dmemextnd':None, 'd2dexnhbrred':None,
# modes_list = {'d2d':None, 'gossip':None, 'hfl':None, 'cfl':None}
# modes_list = {'d2dfullmem':None}
# modes_list = {'d2dgrad': None, 'd2dstalegrad':None, 'd2d':None, 'gossip':None}


def D2DFL(model_type, dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_servers, num_rounds, 
          min_epochs, max_epochs, alpha, skew, overlap, prop, agg_prop, prob_int, prob_ext, stale_lim, cos_lim, args):
    
    print(args)
    print(modes.keys())
    files_list = []
    inter_files_list = []
    # Step 1: Define parameters for the environment, dataset and dataset distribution
    starttime = time.strftime("%Y%m%d_%H%M%S")
    location, num_labels = dataset_approve(dataset)
    
    if model_type == 'shallow':
        if dataset == 'mnist' or dataset == 'fashion':
            in_ch = 1
        elif dataset  == 'cifar':
            in_ch = 3
        base_model = Net(num_labels, in_ch, dataset)
    
    else:
        in_ch = 3
        if model_type == 'vgg16':
            base_model = models.vgg16(num_classes = num_labels)
    
        elif model_type ==  'alexnet':
            base_model = models.alexnet()
            base_model.classifier[6] = nn.Linear(in_features=4096, out_features= num_labels, bias=True)
            
        elif model_type == 'resnet':
            base_model = models.resnet18()
            base_model.fc = nn.Linear(in_features=512, out_features=num_labels, bias=True)
    
        #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata, testidx, validx = dataset_select(dataset, location, in_ch)


    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist. Alpha value will take precedece over skew    
    if skew == 0:
        train_dist = data_noniid(traindata, num_labels, num_nodes, alpha)
    else:
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew)
    
    # Uniform Test distribution for each node. The testing may be carried out on the entire datset

    test_dist = data_noniid(testdata, num_labels, num_nodes, alpha = 10000, idxlist= testidx)
    # Step 4: Create Environment
    env = system_model(num_nodes, num_clusters, num_servers, prob_int= prob_int, prob_ext=prob_ext)
    
    # Create Base Parameter Dictionary for Modes
    base_params = { 'dataset' : dataset, 'num_epochs' : max_epochs, 'num_rounds' : num_rounds, 
                   'num_nodes' : num_nodes, 'base_model' : base_model,'num_labels' : num_labels, 
                   'in_channels' : in_ch, 'traindata' : traindata, 'traindata_dist' : train_dist, 
                   'testdata' : testdata, 'testdata_dist' : test_dist, 'valdata_dist':validx,
                   'batch_size' : batch_size, 'nhood' : env.neighborhood_map, 'env_Lp' : env.Lp, 
                   'num_clusters' : num_clusters, 'num_servers': env.num_servers}
    
    # Flags will only be used for the modes defined at the outset of the main file
    # modes_list = {'d2dstalemodel':None, 'd2dlay':None, 'd2dsel':None, 'd2dstalelay':None, 'd2d':None}
    d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dsel_flags = {'d2d_agg_flg' : 'Sel_D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dlay_flags = {'d2d_agg_flg': 'Layer_D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dstalelay_flags = {'d2d_agg_flg': 'Stale_Layer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dstalemodel_flags = {'d2d_agg_flg': 'Stale_Model', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dgrad_flags = {'d2d_agg_flg': 'grad_ranking', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dstalegrad_flags = {'d2d_agg_flg': 'stalegrad_ranking', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2daccwtexp_flags = {'d2d_agg_flg' : 'D2DAccwtexp', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2daccwtnorm_flags = {'d2d_agg_flg' : 'D2DAccwtnorm', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2daccwtskew_flags = {'d2d_agg_flg' : 'D2DAccwtskew', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2daccwtlin_flags = {'d2d_agg_flg' : 'D2DAccwtlin', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dacc_flags = {'d2d_agg_flg' : 'D2DAcc', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dloss_flags = {'d2d_agg_flg' : 'D2DLoss', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dmem_flags = {'d2d_agg_flg' : 'D2DMem', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dfullmem_flags = {'d2d_agg_flg' : 'D2DFullMem', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}

    d2dexnhbr_flags = {'d2d_agg_flg' : 'D2DExNh', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dexnhbrred_flags = {'d2d_agg_flg' : 'D2DExNhRed', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False} 
    d2dexrly_flags = {'d2d_agg_flg' : 'D2DExRly', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dmemextnd_flags = {'d2d_agg_flg' : 'D2DMemExtnd', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dmemextndrly_flags = {'d2d_agg_flg' : 'D2DMemExtndRly', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}

    d2dmgs_flags = {'d2d_agg_flg' : 'D2DMGS', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    # hd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    hfl_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    # chd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    # intch_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    # intchd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    # intchgossip_flags = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    # intchd2dsel_flags = {'d2d_agg_flg' : 'Layer_D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    gossip_flags = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    # hgossip_flags = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    cfl_flags = {'d2d_agg_flg' : 'CServer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}

    flag_dict = {'d2d': d2d_flags, 'd2dsel': d2dsel_flags, 'd2dlay': d2dlay_flags, 'd2dstalelay': d2dstalelay_flags, 'd2dstalemodel':d2dstalemodel_flags, 
    'd2dgrad': d2dgrad_flags, 'd2dstalegrad': d2dstalegrad_flags, 'gossip':gossip_flags, 'd2daccwtexp':d2daccwtexp_flags, 'd2daccwtnorm':d2daccwtnorm_flags,
    'd2daccwtskew':d2daccwtskew_flags, 'd2daccwtlin':d2daccwtlin_flags,'d2dacc':d2dacc_flags, 'd2dloss':d2dloss_flags, 'd2dmem':d2dmem_flags, 'd2dfullmem':d2dfullmem_flags, 
    'd2dexnhbr': d2dexnhbr_flags, 'd2dexrly': d2dexrly_flags, 'd2dmemextnd':d2dmemextnd_flags, 'd2dmemextndrly':d2dmemextndrly_flags, 'd2dexnhbrred':d2dexnhbrred_flags,
    'd2dmgs':d2dmgs_flags, 'hfl':hfl_flags, 'cfl':cfl_flags}
    
    # flag_dict = {'d2d': d2d_flags, 'd2dsel': d2dsel_flags, 'd2dlay': d2dlay_flags, 'staled2d':staled2d_flags, 'hd2d': hd2d_flags, 'hfl': hfl_flags, 
    #              'chd2d':chd2d_flags, 'intch': intch_flags, 'intchd2d':intchd2d_flags, 'intchd2dsel':intchd2dsel_flags,'gossip':gossip_flags, 'intchgossip':intchgossip_flags,
    #              'hgossip':hgossip_flags, 'cfl':cfl_flags, 'sgd':None}
    
    # Step-5: Create Modes and combine mode params and special flags for all modes under mode_params
    mode_params = {mode:None for mode in modes.keys()}
    for mode in modes.keys():
        if flag_dict[mode] != None:
            mode_params[mode] = {**base_params, **flag_dict[mode]}
        else:
            mode_params[mode] = base_params
        mode_params[mode]['name'] = mode
        
    modelist = [mode for mode in modes.keys()]
    for mode in modelist:
        file_args = {'status': None, 'modename': mode.upper(), 'dataset':dataset.upper(), 'num_nodes':num_nodes, 'clusters':num_clusters, 'epochs':max_epochs, 
                     'rounds': num_rounds, 'skew' : skew, 'alpha' : alpha, 'timestart':starttime, 'prop' : prop, 'aggprop' : agg_prop}
        if mode != 'sgd':
            # Creates Nodeset and other attributes for each mode in modes

            modes[mode] = FL_Modes(**mode_params[mode])
            # Check Hierarchical Aggregation Flag
            if modes[mode].hserver_agg_flg == True:
            # Create Hierarchical Servers 
                modes[mode].form_serverset(env.num_servers, num_labels, in_ch, dataset)
            
            # For D2D specific rounds
            # moderounds = {'d2d': num_rounds * 2}
            # if mode == list(moderounds.keys())[0]:
            #     num_rounds = moderounds[mode]

            # Start Federation Protocol
            for rnd in range(num_rounds):
                # Initiate Local Training on models
                print(f'Update Round {rnd}- Mode {mode}')
                modes[mode].update_round(min_epochs,  max_epochs)

                # Perform validation on Locally trained/fine-tuned models
                print(f'Validation Round {rnd}- Mode {mode}')
                modes[mode].val_round()

                #4-Aggregate from neighborhood  using the weights obtained in the previous step
                print(f'Starting Local Aggregation in round{rnd} for mode {mode}')
                modes[mode].cfl_aggregate_round(rnd, prop, modes[mode].d2d_agg_flg)
                # print(f'CFL Dict: RAM {before_memory / 1000:.2f}MB -> after {after_memory / 1000:.2f} MB  GPU {torch.cuda.memory_allocated() / 1024**2} in MB') 
                
                ref_rnd = (rnd // stale_lim) * stale_lim
                # Check for device Aggregations
                if modes[mode].d2d_agg_flg is not None:
                    if modes[mode].d2d_agg_flg == 'D2D':
                        modes[mode].nhood_aggregate_round(agg_prop)
                    elif modes[mode].d2d_agg_flg ==  'Sel_D2D':
                        modes[mode].selective_aggregate_round(agg_prop, rnd, cos_lim)
                    elif modes[mode].d2d_agg_flg == 'Layer_D2D':
                        modes[mode].layerwise_aggregate_round(agg_prop, cos_lim)
                    elif modes[mode].d2d_agg_flg == 'Stale_Layer':
                        modes[mode].stalelayer_aggregate_round(ref_rnd, agg_prop, cos_lim)
                    elif modes[mode].d2d_agg_flg == 'Stale_Model':
                        modes[mode].stalemodel_aggregate_round(ref_rnd, agg_prop, rnd, cos_lim)
                    elif modes[mode].d2d_agg_flg ==  'Random':
                        modes[mode].random_aggregate_round()
                    elif modes[mode].d2d_agg_flg == 'grad_ranking':
                        modes[mode].gradient_ranking_round(ref_rnd, cos_lim)
                        modes[mode].nhood_aggregate_round(agg_prop)
                    elif modes[mode].d2d_agg_flg == 'stalegrad_ranking':
                        modes[mode].gradient_ranking_round(ref_rnd, cos_lim, global_ref = True)
                        modes[mode].nhood_aggregate_round(agg_prop)
                    
                    elif modes[mode].d2d_agg_flg == 'D2DAccwtexp':
                        modes[mode].d2daccwt_aggregate_round(agg_prop, rnd, alpha = 0.3, beta = 0.05, gamma=0.575, eps = 100, wt = 'exp')
                    elif modes[mode].d2d_agg_flg == 'D2DAccwtnorm':
                        modes[mode].d2daccwt_aggregate_round(agg_prop, rnd, alpha = 0.3, beta = 0.05, gamma=0.575, eps = 100, wt = 'normal')
                    elif modes[mode].d2d_agg_flg == 'D2DAccwtskew':
                        modes[mode].d2daccwt_aggregate_round(agg_prop, rnd, alpha = 0.3, beta = 0.05, gamma=0.575, eps = 100, wt = 'skew')
                    elif modes[mode].d2d_agg_flg == 'D2DAccwtlin':
                        modes[mode].d2daccwt_aggregate_round(agg_prop, rnd, alpha = 0.3, beta = 0.05, gamma=0.575, eps = 100, wt = 'lin')

                    elif modes[mode].d2d_agg_flg == 'D2DAcc':
                        modes[mode].d2dval_aggregate_round(rnd, agg_prop, eps = 100, metric = 'valacc')
                    elif modes[mode].d2d_agg_flg == 'D2DLoss':
                        modes[mode].d2dval_aggregate_round(rnd, agg_prop, eps = 100,  metric = 'valloss')
                    
                    elif modes[mode].d2d_agg_flg == 'D2DMem':
                        modes[mode].d2dmem_aggregate_round(agg_prop)
                    elif modes[mode].d2d_agg_flg == 'D2DFullMem':
                        modes[mode].d2dfullmem_aggregate_round(agg_prop)
                    elif modes[mode].d2d_agg_flg == 'D2DMemExtnd':
                        modes[mode].d2dmemextnd_aggregate_round(agg_prop, rnd)
                    elif modes[mode].d2d_agg_flg == 'D2DMemExtndRly':
                        modes[mode].d2dmemextndrly_aggregate_round(agg_prop, rnd)
                    elif modes[mode].d2d_agg_flg == 'D2DExNh':
                        modes[mode].d2dexnhbr_aggregate_round(agg_prop, rnd)
                    elif modes[mode].d2d_agg_flg == 'D2DExRly':
                        modes[mode].d2dextnrly_aggregate_round(agg_prop, rnd)
                    elif modes[mode].d2d_agg_flg == 'D2DExNhRed':
                        modes[mode].d2dexnhbrred_aggregate_round(agg_prop, rnd)
                    elif modes[mode].d2d_agg_flg == 'D2DMGS':
                        modes[mode].nhood_aggregate_round(agg_prop)
                        modes[mode].random_aggregate_round()


                    modes[mode].cfl_aggregate_round(rnd, prop, modes[mode].d2d_agg_flg)

                # 5- Cluster operations: 
                if modes[mode].ch_agg_flg == True:
                    print(f'Entering Cluster Head Aggregation for mode-{mode} in round-{rnd}')
                    for i in range(env.num_clusters):
                        modes[mode].clshead_aggregate_round(env.cluster_heads[i], env.cluster_set[i], agg_prop) # Added 0.9 instead of agg_prop

                if modes[mode].inter_ch_agg_flg == True:
                    modes[mode].inter_ch_aggregate_round(env.cluster_heads)
                    
                # Should not be executed for Clustered D2D-FL
                if modes[mode].hserver_agg_flg == True: 
                    print(f'Entering Hierarchical Aggregation for mode-{mode} in round-{rnd}')
                    for i in range(env.num_servers):
                        assigned_nodes = []
                        for cluster_id in env.server_groups[i]:
                            assigned_nodes += env.cluster_set[cluster_id] 
                        modes[mode].serverset[i].aggregate_clusters(modes[mode].nodeset, assigned_nodes, modes[mode].weights, prop)

                    #Final Server Aggregation
                    modes[mode].serverset[-1].aggregate_servers(modes[mode].serverset[:-1], modes[mode].nodeset)

                # Perform Testing on aggregated models
                modes[mode].test_round(env.cluster_set)

                modes[mode].record_round_models()
                
                # Interim Record
                if rnd % 5 == 0 and rnd != (num_rounds - 1):
                    file_args['status'] = 'inter'
                    folder = './'
                    files_record = inter_files_list
                    save_file(modes[mode], folder, inter_files_list, args, **file_args)
            
            #folder, status, flmode, mode, dataset, num_nodes, num_clusters, num_epochs, num_rounds, prop, agg_prop, skew, alpha, starttime, files_list
            file_args['status'] = 'Final'
            folder = './Results'
            files_record = files_list 
            save_file(modes[mode], folder, files_list, args, **file_args)
                     
            for node in modes[mode].nodeset:
                node.model.to('cpu')

            if hasattr(modes[mode], 'serverset'):
                for server in modes[mode].serverset:
                    server.model.to('cpu')
            modes[mode].cfl_model.to('cpu')
            torch.cuda.empty_cache()
                        
            del modes[mode]            
            gc.collect()

        elif mode == 'sgd':
            modes[mode] = Servers(0, base_model, records = True)
            sgd_optim = optim.SGD(modes[mode].model.parameters(), lr = 0.01, momentum = 0.9)
            sgd_lambda = lambda epoch: 0.004 * epoch
#             sgd_scheduler = torch.optim.lr_scheduler.LambdaLR(sgd_optim, lr_lambda= sgd_lambda)
            sgd_trainloader = DataLoader(traindata, batch_size = 32)
            sgd_testloader =  DataLoader(testdata)
            num_epochs = max_epochs - min_epochs
            for rnd in range(num_rounds):
                node_update(modes[mode].model.cuda(), sgd_optim, sgd_trainloader, modes[mode].avgtrgloss,
                            modes[mode].avgtrgacc, num_epochs)
                loss, acc = test(modes[mode].model, sgd_testloader)
                modes[mode].avgtestloss.append(loss)
                modes[mode].avgtestacc.append(acc)
            
            folder = './Results'
            file_args['status'] = 'Final'
            save_file(modes[mode], folder, files_list, **file_args)
            
            del modes[mode]
            gc.collect()
            
    combine_results('./Results/', files_list, inter_files_list)
    
## Main Function
if __name__ == "__main__":
#     dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist
    mode_state = D2DFL(modeltype, dataset, batch_size, test_batch_size, modes_list,  nodes, clusters, servers, 
                       rounds, epochs_min, epochs_max, alpha, skew, overlap_factor, prop, agg_prop, prob_int, prob_ext, stale, cos_lim, args)