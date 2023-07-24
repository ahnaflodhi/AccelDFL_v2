import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import random
import copy
import heapq
import sys, gc
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from utils import optimizer_to, scheduler_to

class Net(nn.Module):
    def __init__(self, num_classes, in_ch, dataset):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        if dataset == 'mnist' or dataset == 'fashion':
            self.fc1 = nn.Linear(2000, 1000)
            self.fc2 = nn.Linear(1000, num_classes)
        elif dataset == 'cifar':
            self.fc1 = nn.Linear(2880, 1440)
            self.fc2 = nn.Linear(1440, num_classes)
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_ch, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.1)
#         if dataset == 'mnist' or dataset == 'fashion':
#             self.fc1 = nn.Linear(9216, 128)
#             self.fc2 = nn.Linear(128, num_classes)
#         elif dataset == 'cifar':
#             self.fc1 = nn.Linear(12544, 128)
#             self.fc2 = nn.Linear(128, num_classes)
            
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    @staticmethod
    def add_noise(model, mean = [0.0], std = [0.01]):
        if next(model.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
            
        norm_dist = torch.distributions.Normal(loc = torch.tensor(mean), scale = torch.tensor(std))
        for layer in model.state_dict():
            if 'weight' in layer:
                x = model.state_dict()[layer]
                t = norm_dist.sample((x.view(-1).size())).reshape(x.size()).to(device)
                model.state_dict()[layer].add_(t)
        return model
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
# self.model, self.opt, self.trainset, self.trainloader, self.trgloss, self.trgacc, num_epochs   

def node_update(client_model, optimizer, train_loader, record_loss, record_acc, grad_stats:dict, gradnorms, num_epochs):
#     optimizer_to(optimizer, 'cuda')
#     scheduler_to(scheduler, 'cuda')
    client_model.train()
    for epoch in range(num_epochs):
#         epoch_loss = 0.0
        batch_loss = []
        correct_state = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            total_norm = 0.0
            correct = 0
            data = data.float()
            data, targets = data.cuda(), targets.cuda()
            optimizer.zero_grad()
            # with amp.autocast():
            output = client_model(data)
            loss = F.cross_entropy(output, targets)
            _ , output = torch.max(output.data, 1)
            loss.backward()
            # for p in client_model.parameters():
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** 0.5
            # gradnorms.append(total_norm)
            # grads = [param.grad for param in client_model.parameters()]
            # for i, grad in enumerate(grads):
            #     grad_stats['mean'].append(grad.mean()) # Append Mean
            #     grad_stats['std'].append(grad.std()) # Append std
            # grad_values = [param.grad for param in client_model.parameters()] # Extract grad values as numpy array
            # flat_grads = torch.cat([grad.view(-1) for grad in grads]) # flatten the gradients into a 1D array
            # grad_values = flat_grads.detach().to('cpu').numpy() # convert the flattened gradients to a NumPy array
            # grad_stats['norm_dist'].append(scipy.stats.norm.fit(grad_values)) # Fit Norm Distribution and append paramters.
            optimizer.step()
            correct += (output == targets).float().sum() / output.shape[0]
            batch_loss.append(loss.item())
            correct_state.append(correct.item())

#             if batch_idx % 100 == 0:    # print every 100 mini-batches
#                 print('[%d, %5d] loss-acc: %.3f - %.3f' %(epoch+1, batch_idx+1, sum(batch_loss)/len(batch_loss), sum(correct_state)/len(correct_state)))
#         if epochs_done > 0 and epochs_done % 20 == 0: # Reduce LR after this many epocs.
#             scheduler.step()  
        epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_acc = sum(correct_state) / len(correct_state)
        record_loss.append(round(epoch_loss, 3))
        record_acc.append(round(epoch_acc,3))
#     del data, targets, batch_loss
#     del loss, output, correct_state
#     del epoch_loss, epoch_acc
#     gc.collect()
    
def aggregate(model_list, node_list:list, scale:dict, noise = False):
    agg_model = copy.deepcopy(model_list[0].prevmodel)
    
    # Zeroing container model so that scaling weights may be assigned to each participating model
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)
    
    if noise == True: # Create copies so that original models are not corrupted. Only received ones become noisy
        models = {node:Net.add_noise(copy.deepcopy(model_list[node].prevmodel)) for node in node_list}
        for layer in agg_model.state_dict().keys():
            for node in node_list:
                agg_model.state_dict()[layer].add_(torch.mul(models[node].state_dict()[layer], scale[node]))
            agg_model.state_dict()[layer].div_(len(node_list))
        del models
        gc.collect()
        
    else: # Without adding Noise
        for layer in agg_model.state_dict().keys():
            for node in node_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevmodel.state_dict()[layer], scale[node]))
            agg_model.state_dict()[layer].div_(len(node_list))
        
    return agg_model
# nodeset, agg_targets, self.neighborhood, self.agg_record, scale


def aggregate_reduce(model_list, prev_list:list, red_list:list, scale:dict):
    agg_model = copy.deepcopy(model_list[0].model)

    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)
    
    nodes = list(set(prev_list + red_list))
    
    for layer in agg_model.state_dict().keys():
        for node in nodes:
            agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevmodel.state_dict()[layer], scale[node]))
            model_list[node].selection_counts['agg'] += 1
            if node in red_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].red_model.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['red'] += 1
        agg_model.state_dict()[layer].div_(len(prev_list) + len(red_list))

    return agg_model

def aggregate_tristatus(model_list, prev_list:list, prevrnd_list:list, red_list:list, scale:dict):
    agg_model = copy.deepcopy(model_list[0].model)

    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)
    
    nodes = list(set(prev_list + red_list + prevrnd_list))
    
    for layer in agg_model.state_dict().keys():
        for node in nodes:
            if node in prev_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevmodel.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['agg'] += 1
            elif node in prevrnd_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevrnd_model.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['mem'] += 1
            elif node in red_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].red_model.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['red'] += 1
                
        agg_model.state_dict()[layer].div_(len(prev_list) + len(red_list))

    return agg_model

def aggregate_frommem(model_list, node_list:list, prev_list:list, scale, noise=False):
    agg_model = copy.deepcopy(model_list[0].model)
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)

    nodes = list(set(node_list + prev_list))
    for layer in agg_model.state_dict().keys():
        for node in nodes:
            if node in node_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevmodel.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['agg'] += 1
            elif node in prev_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevrnd_model.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['mem'] += 1

        agg_model.state_dict()[layer].div_(len(node_list) + len(prev_list))
    return agg_model


def aggregate_fullmem(model_list, node_list:list, mem_list:list, mem_array:dict, scale, noise=False): # node_list and prev_list must be mutually exclusive
    agg_model = copy.deepcopy(model_list[0].model)
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)

    if list(mem_array.keys()) != []:
        for node in mem_list:
            if mem_array[node] != None:
                mem_array[node].cuda()

    nodes = list(set(node_list + mem_list))
    for layer in agg_model.state_dict().keys():
        for node in nodes:
            if node in node_list:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevmodel.state_dict()[layer], scale[node]))
                model_list[node].selection_counts['agg'] += 1
            elif node in mem_list:
                agg_model.state_dict()[layer].add_(torch.mul(mem_array[node].state_dict()[layer], scale[node]))
                model_list[node].selection_counts['mem'] += 1

        agg_model.state_dict()[layer].div_(len(node_list) + len(mem_list))

    if list(mem_array.keys()) != []:
        for node in mem_list:
            if mem_array[node] != None:
                mem_array[node].cpu()
    
    return agg_model


def aggregate_prevrnd(model_list, prev_list, scale):
    agg_model = copy.deepcopy(model_list[0].model)
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)

    for layer in agg_model.state_dict().keys():
        for node in prev_list:
            agg_model.state_dict()[layer].add_(torch.mul(model_list[node].prevrnd_model.state_dict()[layer], scale[node]))
            model_list[node].selection_counts['mem'] += 1
        agg_model.state_dict()[layer].div_(len(prev_list))
    
    return agg_model

def selective_aggregate(model_list, agg_full:list, scale:dict, agg_conv = None, noise = False):
    agg_model = copy.deepcopy(model_list[0].model)
    # Zeroing container model so that scaling weights may be assigned to each participating model
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)
    
    if noise == True: # Create copies so that original models are not corrupted. Only received ones become noisy
        model_list = {node:Net.add_noise(copy.deepcopy(model_list[node].model)) for node in (agg_full + agg_conv)}
        
    for layer in agg_model.state_dict().keys():
        if 'weight' in layer:
            for node in agg_full:
                agg_model.state_dict()[layer].add_(torch.mul(model_list[node].model.state_dict()[layer], scale[node]))
        agg_model.state_dict()[layer].div_(len(agg_full))
    
    if len(agg_conv) > 0:
        for layer in agg_model.state_dict().keys():
            if 'conv' in layer and 'weight' in layer:
                for node in agg_conv:
                    agg_model.state_dict()[layer].add_(torch.mul(model_list[node].model.state_dict()[layer], scale[node]))
            agg_model.state_dict()[layer].div_(len(agg_conv))
                
    return agg_model

def layerwise_aggregate(model_list, self_idx, nodelist, scale, thresh, noise = False):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    agg_model = copy.deepcopy(model_list[0].model)
    # Zeroing container model so that scaling weights may be assigned to each participating model
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)
        
    if noise == True: # Create copies so that original models are not corrupted. Only received ones become noisy
        self_model = copy.deepcopy(model_list[self_idx].model)
        model_list = {node:Net.add_noise(copy.deepcopy(model_list[node].model)) for node in nodelist}
        model_list[self_idx] = self_model

    #Append self model after noise has been added
    nodelist.append(self_idx)
    for layer in agg_model.state_dict().keys():
        if 'weight' in layer:
            ref = model_list[self_idx].model.state_dict()[layer].view(-1).detach()
            i = 0
            for node in nodelist:
                tgt = model_list[node].model.state_dict()[layer].view(-1).detach()
                cos_val = cos(ref, tgt).item()
#                 print(layer, cos_val, flush = True, end = ',')
                if cos_val >= thresh:
                    agg_model.state_dict()[layer].add_(torch.mul(model_list[node].model.state_dict()[layer], scale[node]))
                    i += 1
            agg_model.state_dict()[layer].div_(i)
    if noise == True:
        del model_list
    del ref
    del tgt
    gc.collect()

    return agg_model

def stalelayerwise_aggregate(model_list, self_idx, stale_cflmodel, nodelist, scale, thresh, noise = False):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    agg_model = copy.deepcopy(model_list[0].model)
    # Zeroing container model so that scaling weights may be assigned to each participating model
    for layer in agg_model.state_dict().keys():
        agg_model.state_dict()[layer].mul_(0.00)
        
    if noise == True: # Create copies so that original models are not corrupted. Only received ones become noisy
        self_model = copy.deepcopy(model_list[self_idx].model)
        model_list = {node:Net.add_noise(copy.deepcopy(model_list[node].model)) for node in nodelist}
        model_list[self_idx] = self_model

    #Append self model after noise has been added    
    nodelist.append(self_idx)
    for layer in agg_model.state_dict().keys():
        if 'weight' in layer:
            ref = stale_cflmodel.state_dict()[layer].view(-1).detach().cuda()
            i = 0
            for node in nodelist:
                tgt = model_list[node].model.state_dict()[layer].view(-1).detach()
                if node != self_idx:
                    cos_val = cos(ref, tgt).item()
                    if cos_val >= thresh:
                        agg_model.state_dict()[layer].add_(torch.mul(model_list[node].model.state_dict()[layer], scale[node]))
                        i += 1
                else:
                    agg_model.state_dict()[layer].add_(torch.mul(model_list[self_idx].model.state_dict()[layer], scale[self_idx]))
                    i +=1 

            agg_model.state_dict()[layer].div_(i)
            
    return agg_model

def model_checker(model1, model2):
    models_differ = 0
    for modeldata1, modeldata2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(modeldata1[1], modeldata2[1]):
            pass
        else:
            models_differ += 1
            if (modeldata1[0] ==  modeldata2[0]):
                print("Mismatch at ", modeldata1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

def test_gradients(model, test_loader):
    grad_vectors = np.zeros((10))
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        loss.backward()
        grad_vectors += torch.sum(model.fc2.weight.grad, dim = 1).to('cpu').numpy()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
#         print(f' Prediction {pred.item()}, target{target.item()}  Correct so far {correct}')
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    # grad_vectors  = grad_vectors / np.linalg.norm(grad_vectors)
    return grad_vectors

def extract_weights(model, add_noise = True):
    weights = {}
    for key in model.state_dict():
        if 'weight' not in key:
            continue
        weights[key] = model.state_dict()[key]
    return weights

def calculate_divergence(modes, main_model_dict, cluster_set, num_nodes, divergence_results):
    centr_fed_ref = np.random.randint(0, num_nodes)
    for mode in modes:
        basemodel_keys = main_model_dict[mode][0].state_dict().keys()
        break
     # Structure of Dictionary   
    # divergence_results {mode: {node:{layer:[divergence for each round]}}       
                                      
    ref_model = main_model_dict['SGD']
    ref_weight = extract_weights(ref_model)
    
    for mode in modes:
        if mode != 'SGD':
            for target_node in range(num_nodes):
                target_model = main_model_dict[mode][target_node].cuda()
                target_weight = extract_weights(target_model)
                for layer in ref_weight.keys():                          
                    divergence_results[mode][target_node][layer].append(torch.linalg.norm(ref_weight[layer] - target_weight[layer]))
    return  divergence_results

def clustering_divergence(model_dict, cluster_graph, num_nodes):
    divergence_results = []
    neighborhood = []
    basemodel_keys = model_dict[0].state_dict().keys()
    div_recorder = {}
    div_recorder_conv = {}
    div_recorder_fc = {}
    
    for node in range(num_nodes):
        temp = [neighbor for neighbor in cluster_graph.neighbors(node)]
        neighborhood.append(temp)
        div_recorder[node] = {neighbor:None for neighbor in temp}
        div_recorder_conv[node] = {neighbor:None for neighbor in temp}
        div_recorder_fc[node] = {neighbor:None for neighbor in temp}
        
        
    for ref_node, neighbor_nodes in enumerate(neighborhood):
        for neighbor in neighbor_nodes:
            total_diff = 0
            conv_diff = 0
            fc_diff = 0
            for layer in model_dict[ref_node].state_dict():
                if 'weight' not in layer:
                    continue
                diff = torch.linalg.norm(model_dict[ref_node].state_dict()[layer] - model_dict[neighbor].state_dict()[layer])
                total_diff += diff.item()
                if 'conv' in layer:
                    conv_diff += diff.item()
                elif 'fc' in layer:
                    fc_diff += diff.item()
                
            div_recorder[ref_node][neighbor] = total_diff
            div_recorder_conv[ref_node][neighbor] = conv_diff
            div_recorder_fc[ref_node][neighbor] = fc_diff
    
     #Normalize
    div_recorder = normalize_div(div_recorder)
    div_recorder_conv = normalize_div(div_recorder_conv)
    div_recorder_fc = normalize_div(div_recorder_fc)
            
    return div_recorder, div_recorder_conv, div_recorder_fc

def normalize_div(div_dict):
    for node in range(len(div_dict)):
        temp = []
        print(div_dict[node])
        for _ , val in div_dict[node].items():
            temp.append(val)
            norm_factor = np.linalg.norm(temp)
            
        for neighbor, _ in div_dict[node].items():
            div_dict[node][neighbor] = div_dict[node][neighbor] / norm_factor
    return div_dict

def revise_neighborhood(div_dict, n, sort_type = 'min'):
    revised_neighborhood = []
    for node in range(len(div_dict)):
        if sort_type == 'min':
            temp = heapq.nsmallest(n, div_dict[node].items() , key=lambda i: i[1])
        elif sort_type == 'max':
            temp = heapq.nlargest(n, div_dict[node].items() , key=lambda i: i[1])
        temp_nodes = [max_node[0] for max_node in temp]
        revised_neighborhood.append(temp_nodes)
    return revised_neighborhood
    
                     