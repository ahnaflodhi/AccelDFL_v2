# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

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
zeta = args.a
stale = args.D
cos_lim = args.sim
prob_int = args.clint
prob_ext = args.clext

modes_list = {'d2daccep':None}


def D2DFL(model_type, dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_servers, num_rounds, 
          min_epochs, max_epochs, zeta, skew, overlap, prop, agg_prop, prob_int, prob_ext, stale_lim, cos_lim, args):
    
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
    traindata, testdata = dataset_select(dataset, location, in_ch)

    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist. Alpha value will take precedece over skew    
    if skew == 0:
        train_dist = data_noniid(traindata, num_labels, num_nodes, zeta)
    else:
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew)
    
    # Uniform Test distribution for each node. The testing may be carried out on the entire datset
    test_dist = data_iid(testdata, num_labels, num_nodes)
    
    # Step 4: Create Environment
    env = system_model(num_nodes, num_clusters, num_servers, prob_int= prob_int, prob_ext=prob_ext)
    
    # Create Base Parameter Dictionary for Modes
    base_params = { 'dataset' : dataset, 'num_epochs' : max_epochs, 'num_rounds' : num_rounds, 
                   'num_nodes' : num_nodes, 'base_model' : base_model,'num_labels' : num_labels, 
                   'in_channels' : in_ch, 'traindata' : traindata, 'traindata_dist' : train_dist, 
                   'testdata' : testdata, 'testdata_dist' : test_dist, 'batch_size' : batch_size,
                   'nhood' : env.neighborhood_map, 'env_Lp' : env.Lp, 'num_clusters' : num_clusters,
                   'num_servers': env.num_servers}
    
    # Flags will only be used for the modes defined at the outset of the main file
    # modes_list = {'d2dstalemodel':None, 'd2dlay':None, 'd2dsel':None, 'd2dstalelay':None, 'd2d':None}
    d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dsel_flags = {'d2d_agg_flg' : 'Sel_D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    d2dlay_flags = {'d2d_agg_flg': 'Layer_D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dstalelay_flags = {'d2d_agg_flg': 'Stale_Layer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dstalemodel_flags = {'d2d_agg_flg': 'Stale_Model', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dgrad_flags = {'d2d_agg_flg': 'grad_ranking', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2dstalegrad_flags = {'d2d_agg_flg': 'stalegrad_ranking', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False }
    d2daccep_flags = {'d2d_agg_flg' : 'D2DAcc', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    
        
    # hd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    # hfl_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    # chd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    # intch_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    # intchd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    # intchgossip_flags = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    # intchd2dsel_flags = {'d2d_agg_flg' : 'Layer_D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    gossip_flags = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    # hgossip_flags = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    # cfl_flags = {'d2d_agg_flg' : 'CServer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}

    flag_dict = {'d2d': d2d_flags, 'd2dsel': d2dsel_flags, 'd2dlay': d2dlay_flags, 'd2dstalelay': d2dstalelay_flags, 'd2dstalemodel':d2dstalemodel_flags, 
    'd2dgrad': d2dgrad_flags, 'd2dstalegrad': d2dstalegrad_flags, 'gossip':gossip_flags, 'd2daccep':d2daccep_flags}
    
    
    # Step-5: Create Modes and combine mode params and special flags for all modes under mode_params
    mode_params = {mode:None for mode in modes.keys()}
    for mode in modes.keys():
        if flag_dict[mode] != None:
            mode_params[mode] = {**base_params, **flag_dict[mode]}
        else:
            mode_params[mode] = base_params
        mode_params[mode]['name'] = mode
        
        file_args = {'status': None, 'modename': mode.upper(), 'dataset':dataset.upper(), 'num_nodes':num_nodes, 'clusters':num_clusters, 'epochs':max_epochs, 
                     'rounds': num_rounds, 'skew' : skew, 'alpha' : zeta, 'timestart':starttime, 'prop' : prop, 'aggprop' : agg_prop}
        
    
    best = {'alpha':None, 'beta':None, 'gamma':None, 'testacc':0.0, **file_args}

    alphas = np.round(np.linspace(0.5, 0.10, 5), 3)
    betas = np.round(np.linspace(0.05, 0.2, 5), 3)
    gammas = np.round(np.linspace (0.2, 0.7, 5), 3)
    
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                modes['d2daccep'] = FL_Modes(**mode_params['d2daccep'])
                # Check Hierarchical Aggregation Flag
                if modes[mode].hserver_agg_flg == True:
                # Create Hierarchical Servers 
                    modes[mode].form_serverset(env.num_servers, num_labels, in_ch, dataset)
                # Start Federation Protocol
                for rnd in range(num_rounds):
                    # Initiate Local Training on models
                    print(f'Update Round {rnd}- Mode {mode}')
                    modes[mode].update_round(min_epochs,  max_epochs)

                    # Perform Testing on Locally trained/fine-tuned models
                    modes[mode].test_round(env.cluster_set)
                    #4-Aggregate from neighborhood  using the weights obtained in the previous step
                    print(f'Starting Local Aggregation in round{rnd} for mode {mode}')
                    modes[mode].cfl_aggregate_round(rnd, prop, modes[mode].d2d_agg_flg)
                    # print(f'CFL Dict: RAM {before_memory / 1000:.2f}MB -> after {after_memory / 1000:.2f} MB  GPU {torch.cuda.memory_allocated() / 1024**2} in MB') 

                    modes[mode].d2dacc_aggregate_round(agg_prop, rnd, alpha, beta, gamma)
                
                if modes[mode].testacc[-1] >= best['testacc']:
                    best['testacc'] = modes[mode].testacc[-1]
                    best['alpha'] = alpha
                    best['beta'] = beta
                    best['gamma']  = gamma
                
                print('accuracy:{}, alpha:{}, beta:{}, gamma:{}'.format(best['testacc'], best['alpha'], best['beta'], best['gamma']))
                torch.cuda.empty_cache()
                
                del modes[mode]            
                gc.collect()

    
    filename = 'gridsearch_res' + '_'+ starttime
    with open(filename, 'wb') as final:
        pickle.dump(best, final)
    
    for node in modes[mode].nodeset:
        node.model.to('cpu')

    if hasattr(modes[mode], 'serverset'):
        for server in modes[mode].serverset:
            server.model.to('cpu')
    modes[mode].cfl_model.to('cpu')
    

            
    # combine_results('./Results/', files_list, inter_files_list)
    
## Main Function
#     dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist
mode_state = D2DFL(modeltype, dataset, batch_size, test_batch_size, modes_list,  nodes, clusters, servers, 
                    rounds, epochs_min, epochs_max, zeta, skew, overlap_factor, prop, agg_prop, prob_int, prob_ext, stale, cos_lim, args)