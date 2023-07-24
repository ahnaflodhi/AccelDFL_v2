import random
import torch
import pickle
import os

def constrained_sum(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur.
    """
    # divider = []
    # while 1 in divider or len(divider) == 0:
    #     dividers = sorted(random.sample(range(1, total), n - 1))
    #     divider = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    # return divider

    # Alternate method: Less Random albeit rapid solution
    if n <= total:
        try:
            rem = total % n
            divider = [total // n ] * n
            for i in range(rem):
                divider[i] +=  1
        except:
            print('Division of clusters among servers didnt work as servers more than clusters')
    return divider


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
#     print('model size: {:.3f}MB'.format(model_mb))
    return model_mb

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


# def save_file(folder, status, flmode, mode, dataset, num_nodes, num_clusters, num_epochs, num_rounds, prop, agg_prop, skew, alpha, starttime, files_list):
def save_file(flmode, folder, files_list, args, **nameargs):
    file_name = ''
    for i, (namespec, namevals) in enumerate(nameargs.items()):
        if i == 0:
            file_name += namevals
        else:
            file_name += '_' + namespec[0] + str(namevals)

#     file_name = status + '_' +str(mode).upper() + '_' + dataset.upper() + '_' +'s'+ str(skew) + '_' + 'a' + str(alpha)  + '_' +'n'+ str(num_nodes)  + '_' + 'c' + str(num_clusters) + '_' +'e' + str(num_epochs) + '_' + 'r' + str(num_rounds) + '_' + 'prop' + str(prop) + '_' + 'aggprop' + str(agg_prop) +'_' +  starttime
    files_list.append(file_name) # 
    file_name = os.path.join(folder, file_name)
    saved_set = {}
    if nameargs['modename'] != 'sgd':
        saved_set = {'avgtrgloss' : flmode.avgtrgloss,
                      'avgtrgacc' : flmode.avgtrgacc,
                      'avgtestloss' : flmode.avgtestloss,
                      'avgtestacc' : flmode.avgtestacc,
                      'cluster_trgloss' : flmode.cluster_trgloss,
                      'cluster_trgacc' : flmode.cluster_trgacc,
                      'cluster_testloss' : flmode.cluster_testloss,
                      'cluster_testacc' : flmode.cluster_testacc,
                      'nodetrgloss' : {node: flmode.nodeset[node].trgloss for node in range(nameargs['num_nodes'])},
                      'nodetrgacc' : {node:flmode.nodeset[node].trgacc for node in range(nameargs['num_nodes'])},
                      'nodetestloss' : {node:flmode.nodeset[node].testloss for node in range(nameargs['num_nodes'])},
                      'nodetestacc' : {node:flmode.nodeset[node].testacc for node in range(nameargs['num_nodes'])},
                      'divergence_dict' : {node:flmode.nodeset[node].divergence_dict for node in range(nameargs['num_nodes'])},
                      'divegence_cov_dict' : {node:flmode.nodeset[node].divergence_conv_dict for node in range(nameargs['num_nodes'])},
                      'divergence_fc_dict' : {node:flmode.nodeset[node].divergence_fc_dict for node in range(nameargs['num_nodes'])},
                      'neighborhood' : {node:flmode.nodeset[node].neighborhood for node in range(nameargs['num_nodes'])},
                      'ranked_nhood' : {node:flmode.nodeset[node].ranked_nhood for node in range(nameargs['num_nodes'])},
                      'node_degree' : {node:flmode.nodeset[node].degree for node in range(nameargs['num_nodes'])},
                      'cos_vals' : {node:flmode.nodeset[node].cos_vals for node in range(nameargs['num_nodes'])},
                      'global_cosvals' : {node:flmode.nodeset[node].global_cosvals for node in range(nameargs['num_nodes'])},
                      'label_data':{node:flmode.nodeset[node].label_data for node in range(nameargs['num_nodes'])},
                      'gradnorms':{node:flmode.nodeset[node].gradnorms for node in range(nameargs['num_nodes'])},
                      'selection_counts':{node:flmode.nodeset[node].selection_counts for node in range(nameargs['num_nodes'])},
                      'arguments': args
                      } # 
                      #'grad_stats':{node:flmode.nodeset[node].grad_stats for node in range(nameargs['num_nodes'])},
    elif nameargs['modename'] == 'sgd':
        saved_set = {'avgtrgloss' : flmode.avgtrgloss,
                      'avgtrgacc' : flmode.avgtrgacc,
                      'avgtestloss' : flmode.avgtestloss,
                      'avgtestacc' : flmode.avgtestacc}
        
    final_set = {nameargs['modename']: saved_set}
    with open(file_name, 'wb') as ffinal:
        pickle.dump(final_set, ffinal)
        
#     if nameargs['status'] == 'Final':
#         inter_file = 'inter' + '_' + str(modename).upper() + '_' + dataset.upper() + '_' + dist.upper()  + '_' +'n'+ str(num_nodes)  + '_' + 'c' + str(num_clusters) + '_' +'e' + str(num_epochs) + '_' + 'r' + str(num_rounds) + '_' + starttime
#         if inter_file in os.listdir():
#             try:
#                 os.remove(inter_file)
#                 print('Removed Intermediate Results')
#             except:
#                 print('Cannot Locate the intermediate results file')
#         else:
#             print('Cannot locate file')
          
def combine_results(folder, files_list, inter_list):
    joint_set = {}
    for file in files_list:
        with open(os.path.join(folder,file), 'rb') as f:
            state = pickle.load(f)
        for mode, records in state.items():
            joint_set[mode] = records
    
    jointfile = '_'.join(files_list[0].split('_')[2:])
    jointfile = os.path.join(folder, jointfile)
    with open(jointfile, 'wb') as f:
        pickle.dump(joint_set, f)
        
    for file in files_list:
        os.remove(os.path.join(folder, file))
        
    for file in inter_list:
        os.remove(os.path.join('./', file))
        
    print('Finished Combining and removing interim results')
