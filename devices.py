from DNN import *
# import heapq, resource
import numpy as np
import math
import scipy.stats as stats

from data_utils import DataSubset
import copy, gc

import torch
# import torchvision
# import torch.optim as optimss
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader #, Dataset, TensorDataset, IterableDataset

class Nodes:
    """
    Generates node status and recording dictionaries
    """
    #idx, self.base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, self.dataset, self.batch_size, node_n_nhood
    def __init__(self, node_idx: int, base_model, num_labels: int, in_channels: int, 
                 traindata, trg_dist:list, testdata, test_dist:list, dataset:str, batch_size:int,
                 node_neighborhood: list, lr = 0.01, wt_init = False, role = 'node'):
        """
        Creates the Node Object.
        Contains methods for individual nodes to perform.
        Requires neighborhood information for each node.
        """
        #Node properties
        self.idx = node_idx
        self.batch_size = batch_size
        self.neighborhood = node_neighborhood
        self.ranked_nhood = node_neighborhood
        self.degree = len(self.neighborhood)
        self.role = role
        self.epochs = 0 # Time of creation -Assuming no learning has taken place. Necessary for LR scheduler
        self.lr = lr # LR employed by node
        
        # Dataset and data dist related
        self.trainset = trg_dist[self.idx]
        self.trainloader = DataLoader(DataSubset(traindata, trg_dist, self.idx), batch_size = batch_size)
        self.label_data = traindata.targets[trg_dist[self.idx]].unique()
        # print(self.label_data, flush= True, end= '-')
        self.testset = test_dist[self.idx]
        self.testloader = DataLoader(DataSubset(testdata, test_dist, self.idx))
        self.grad_testloader = DataLoader(DataSubset(testdata, test_dist, self.idx), batch_size = 8)

        self.base_model_selection(base_model, num_labels, in_channels, dataset, wt_init)
        
        # Recorders
        self.trgloss = []
        self.trgacc = []
        self.testloss = []
        self.testacc = []
        self.valloss = []
        self.valacc = []
        self.average_epochloss = 0
        
        # Appending self-idx to record CFL divergence
        # Divergence Targets
        div_targets = self.neighborhood
        self.agg_record = [] # Nodes aggregated in the previous round
        self.cos_vals = {nhbr:[] for nhbr in self.neighborhood}
        self.neighbor_grads = {i:[] for i in (self.neighborhood)}
        self.global_cosvals = {nhbr:[] for nhbr in self.neighborhood}
        self.gradnorms = []
        self.grad_stats = {'mean':[], 'std':[], 'norm_dist':[]}
        self.nhood_acc = {nhbr:[] for nhbr in self.neighborhood}
        self.divergence_dict = {node:[] for node in div_targets}
        self.divergence_conv_dict = {node:[] for node in div_targets}
        self.divergence_fc_dict = {node:[] for node in div_targets}
        
    def base_model_selection(self, base_model, num_labels, in_channels, dataset, wt_init):
        # Same weight initialization
        self.model = copy.deepcopy(base_model)
        self.prevmodel = copy.deepcopy(self.model).to('cuda') # Record of the node's own model from the previous round
#         if wt_init == True:
#             self.model.load_state_dict(base_mode.state_dict())
        self.opt = optim.SGD(self.model.parameters(), lr = self.lr) # , momentum = 0.9
#         lambda_sch = lambda epoch: 1 * epoch
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda= lambda_sch)
 
    def local_update(self, num_epochs):
        if len(self.trainset) > 0: # For extreme alpha cases where a device might not get any data allocated.
            node_update(self.model, self.opt, self.trainloader, self.trgloss, self.trgacc, self.grad_stats, self.gradnorms, num_epochs)
            # print('Mean:{}, SD: {}, Norm: {}'.format(self.grad_stats['mean'][-1], self.grad_stats['std'][-1], self.grad_stats['norm_dist'][-1]))
            self.epochs_in_rnd = num_epochs
            self.epochs += num_epochs
        else:
            self.trgacc.append(0.00)
            self.trgloss.append(0.00)
        # print(f'Node{self.idx}-{self.epochs}', end = ', ', flush = True)
#         if len(self.trgloss) > 1:
#             print(f'Node {self.idx} : Delta Trgloss = {self.trgloss[-2] - self.trgloss[-1]:0.3f}', end = ",  ", flush = True)
#         else:
#             print(f'Node {self.idx}: Trgloss = {self.trgloss[-1]:0.3f}', end = ",  ")

    def node_test(self):
        test_loss, test_acc = test(self.model, self.testloader)
        _, prev_acc = test(self.prevmodel, self.testloader)
        self.testloss.append(test_loss)
        self.testacc.append(test_acc)
        print(f'Node{self.idx}: Test Acc= {self.testacc[-1]:0.3f} Prev Acc = {prev_acc:0.3f}', end = ", ", flush = True) #LR={self.opt.param_groups[0]["lr"]} Trg Loss= {self.trgloss[-1]:0.3f} Trg Acc= {self.trgacc[-1]}

    def node_val(self, valloader):
        val_loss, val_acc = test(self.model, valloader)
        self.valloss.append(val_loss)
        self.valacc.append(val_acc)
        # print(f'Node{self.idx}: Val Acc= {self.valacc[-1]:0.3f}, Val Loss= {self.valloss[-1]:0.3f}', end = ", ", flush = True)

    def scale_update(self, weightage):
        scale = {node:1.0 for node in self.neighborhood}
        return scale
            
    def aggregate_nodes(self, nodeset, agg_prop, scale:dict, cluster_set = None):
        # Choosing the #agg_count number of highest ranked nodes for aggregation
        # If Node aggregating Nhood
        if cluster_set == None:
            agg_scope = int(np.ceil(agg_prop * len(self.neighborhood)))
            agg_targets = random.sample(self.neighborhood,agg_scope)
            agg_targets.append(self.idx)     
            
        # If CH aggregating Cluster    
        else:
            agg_scope = int(np.ceil(agg_prop * len(cluster_set)))
            if agg_scope >= 1 and agg_scope <= len(cluster_set):
                try:
                    # No need to add self index since cluster-head id already included in cluster-set
                    agg_targets = random.sample(cluster_set, agg_scope)
                    agg_targets.append(self.idx)
                except:
                    print(f'Agg_scope {agg_scope} does not conform to size of Cluster_Set {len(cluster_set)}')

        
        agg_model = aggregate(nodeset, agg_targets, scale)
        
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()

    def aggregate_nodesmemory(self, nodeset, agg_prop, scale:dict):
        agg_scope = int(np.ceil(agg_prop * len(self.neighborhood)))
        agg_targets = random.sample(self.neighborhood,agg_scope)
        
        agg_targets.append(self.idx)
        _, acc = test(self.prevmodel, self.testloader)

        agg_model = aggregate_frommem(nodeset, agg_targets, self.neighborhood, self.agg_record, scale)
        self.model.load_state_dict(agg_model.state_dict())

        self.agg_record = agg_targets # Update aggregation record

        del agg_model
        gc.collect()

    def aggregate_extndnhbr(self, nodeset, agg_prop, rnd, scale):
        agg_scope = int(np.ceil(agg_prop * len(self.neighborhood)))
        agg_targets = random.sample(self.neighborhood,agg_scope) # Receiving step
        if rnd > 0:
            temp = []
            for node in nodeset:
                if node.idx in agg_targets:
                    nonoverlap_nhbr = [x for x in node.agg_record if x not in self.neighborhood]
                    temp.append(random.sample(nonoverlap_nhbr,1)[0])

            agg_targets = list(set(agg_targets + temp))

        agg_targets.append(self.idx) #Adding self
        # print(agg_scope, self.neighborhood , agg_targets)
        agg_model = aggregate(nodeset, agg_targets, scale) # Aggregating
        self.model.load_state_dict(agg_model.state_dict()) # Updating local model

        self.agg_record = agg_targets
        self.agg_record.remove(self.idx)

        del agg_model
        gc.collect()

    def aggregate_random(self, nodeset, scale):
        target_id = self.idx
        while target_id == self.idx:
            target_id = random.sample(list(range(len(nodeset))), 1)[0]
        node_list = [self.idx, target_id]                     
        agg_model = aggregate(nodeset, node_list, scale)
#         self.model = copy.deepcopy(agg_model)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()
    
    @staticmethod
    def vectorize_model(model, key = 'all'):
        temp = []
        for layer in model.state_dict():
            if key != 'all': # Vectorize parts of model
                if 'weight' in layer and key in layer:
                    temp.append(model.state_dict()[layer].view(-1).detach())
            else: # Vectorize complete model
                if 'weight' in layer: 
                    temp.append(model.state_dict()[layer].view(-1).detach())
        x = torch.cat(temp)
        # x = torch.cat([p.flatten() for p in model.parameters()]).detach()
        return x
    
    def cos_check(self, nodeset):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self_vec = self.vectorize_model(self.model, 'all').cpu()
        for neighbor in self.neighborhood:
            temp = self.vectorize_model(nodeset[neighbor].model, key = 'all').cpu()
            # self.cos_vals[neighbor].append(cos(self_vec, temp).item())
            self.cos_vals[neighbor].append(((self_vec @ temp.T)/(np.linalg.norm(self_vec) * np.linalg.norm(temp))).item())
            
        del self_vec
        del temp
        gc.collect()

    def global_coscheck(self, globalmodel, nodeset):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self_vec = self.vectorize_model(globalmodel, 'all').to('cpu')
        for neighbor in self.neighborhood:
            temp = self.vectorize_model(nodeset[neighbor].model, key = 'all').cpu()
            # self.global_cosvals[neighbor].append(cos(self_vec, temp).item())
            self.global_cosvals[neighbor].append(((self_vec @ temp.T)/(np.linalg.norm(self_vec) * np.linalg.norm(temp))).item())

        del self_vec
        del temp
        gc.collect()
    
    def aggregate_selective(self, nodeset, scale, agg_prop, rnd, cos_thresh):
        self.cos_check(nodeset)
        agg_full = []
        agg_conv = []
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        nodelist = random.sample(self.neighborhood, target_nodes)
        # if rnd > 6:
        #     for nhbr in nodelist:
        #         if self.cos_vals[nhbr][-1] >= cos_thresh:
        #             agg_full.append(nhbr)
        #         else:
        #             agg_conv.append(nhbr)
        #     print(f' Node: {self.idx} Agg_full {agg_full}, Agg Conv {agg_conv} Agg_vals {self.cos_vals}', flush = True, end = ',')
        # else:
        #     agg_full = self.neighborhood[:]
        agg_full = sorted(self.cos_vals, key=self.cos_vals.get, reverse=True)[:target_nodes]
        agg_full.append(self.idx)
        
        agg_model = selective_aggregate(nodeset, agg_full, scale, agg_conv = agg_conv)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()
        
    def aggregate_layerwise(self, nodeset, scale, agg_prop, cos_thresh):
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        nodelist = random.sample(self.neighborhood, target_nodes)
        agg_model = layerwise_aggregate(nodeset, self.idx, nodelist, scale, cos_thresh)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()
        
    def stale_aggregate_layerwise(self, nodeset, staleglobal, scale, agg_prop,  cos_thresh):
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        nodelist = random.sample(self.neighborhood, target_nodes)
        agg_model = stalelayerwise_aggregate(nodeset, self.idx, staleglobal,  nodelist, scale, cos_thresh, noise = False)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()

    def d2dval_aggregate(self, nodeset, rnd, eps, agg_prop, scale:dict, key):

        metric_dict, normmetric_dict = self.nhbrhood_dict(nodeset, eps, key = key)
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        
        if rnd % 3 == 0:
            if key == 'valacc':
                agg_targets = sorted(metric_dict, key=metric_dict.get, reverse=True)[:target_nodes]
            elif key == 'valloss':
                agg_targets = sorted(metric_dict, key=metric_dict.get, reverse=False)[:target_nodes]
        else:
            agg_targets = random.sample(self.neighborhood, target_nodes)
        
        agg_targets.append(self.idx)
        agg_model = aggregate(nodeset, agg_targets, scale)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()


    def nhbrhood_dict(self, nodeset, eps = 1000, key = 'valacc'):
        reg_dict = {nhbr:None for nhbr in self.neighborhood}
        norm_dict ={nhbr:None for nhbr in self.neighborhood}

        for node in self.neighborhood:
            if key == 'valacc':
                reg_dict[node] = nodeset[node].valacc[-1]
            elif key == 'valloss':
                reg_dict[node] = nodeset[node].valloss[-1]
 
        max_nhbr = max(reg_dict, key = reg_dict.get)
        min_nhbr = min(reg_dict, key = reg_dict.get)
        for node in self.neighborhood:
            denom = (reg_dict[max_nhbr] - reg_dict[min_nhbr])
            if denom == 0:
                denom = 1.0
            norm_dict[node] = math.exp(eps * ((reg_dict[node] - reg_dict[max_nhbr]) / denom))
        
        # Sorting Just in case
        sorted_regdict = {n:reg_dict[n] for n in sorted(reg_dict, key=reg_dict.get, reverse=True)}
        sorted_normdict = {n:norm_dict[n] for n in sorted(norm_dict, key=norm_dict.get, reverse=True)}

        return reg_dict, norm_dict

    def d2daccwt_aggregate(self, nodeset, rnd, alpha, beta, gamma, eps, wt, scale:dict, agg_prop:float):
        accwt_dict= self.wtd_ranking(nodeset, rnd, alpha, beta, gamma, eps, wt = wt)
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        # agg_targets = sorted(accwt_dict, key=accwt_dict.keys(), reverse = True)[:target_nodes] #Picking highest weights
        
        agg_targets = list(np.random.choice(list(accwt_dict.keys()), size = target_nodes, replace = False, p = list(accwt_dict.values())))

        # #Picking stochastically based on exponential weights
        # agg_targets = []
        # while len(agg_targets) < target_nodes:
        #     sampled_key = random.choices(list(accwt_dict.keys()), weights=list(accwt_dict.values()), k=1)
        #     if sampled_key[0] not in agg_targets:
        #         agg_targets.append(sampled_key[0])
        
        agg_targets.append(self.idx)
        print(agg_targets)
        agg_model = aggregate(nodeset, agg_targets, scale)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()

    def wtd_ranking(self, nodeset, rnd, alpha, beta, gamma, eps, wt = 'exp'): #alpha = 0.3, beta=0.1, gamma1=0.3, gamma2 = 0.03
        rnd += 1
        weights = []
        acc_dict, _ = self.nhbrhood_dict(nodeset, eps, key = 'valacc')
        loss_dict , _ = self.nhbrhood_dict(nodeset, eps, key = 'valloss')


        for node in self.neighborhood:
            
            # Delta Acc and Acc
            # if rnd <= 1:
            #     # delta_factor = np.exp(beta * (nodeset[node].testacc[-1]) / (1 + gamma*rnd))  alpha = 0.1, beta=0.775, gamma=0.3
            #     delta_factor = 1 + np.exp(-(1 + gamma1*rnd) * nodeset[node].testacc[-1]/alpha)
            # else:
            #     delta_factor =  1 + np.exp(-(1 + gamma1*rnd) * (nodeset[node].testacc[-1] - nodeset[node].testacc[-2])/alpha)

            # # acc_factor = 0.3 + np.exp(-alpha * nodeset[node].testacc[-1] / (1 + gamma*rnd)) alpha = 0.1, beta=0.775, gamma=0.3
            # # acc_factor = 0.75 + (1 - np.exp(-rnd * alpha))/2

            # Delta Acc and Acc: Version 1 : alpha = 0.775, beta = 0.1, gamma = 0.3
            if rnd <= 1:
                delta_factor = np.exp(alpha * (nodeset[node].valacc[-1]) / (1 + gamma*rnd))
            else:
                delta_factor =  np.exp(alpha * (nodeset[node].valacc[-1] - nodeset[node].valacc[-2]) / (1 + gamma*rnd))

            acc_factor = 0.3 + np.exp(-beta * nodeset[node].valacc[-1] / (1 + gamma*rnd))


            weights.append(delta_factor * acc_factor)


            # # 28 Mar version
            # alpha = 0.04
            # beta = 0.2
            # gamma1 = 0.3
            # gamma2 = 0.03
            # epsilon = 0.01
            # # Delta Acc and Acc: Version 2
            # if rnd <= 1:
            #     delta_acc = nodeset[node].testacc[-1]
            # else:
            #     delta_acc = nodeset[node].testacc[-1] - nodeset[node].testacc[-2]
            # delta_factor = 1 + np.exp(-(1+gamma1*rnd) * alpha / (epsilon + delta_acc))
            
            # acc_factor = 1 + np.exp(beta / (nodeset[node].testacc[-1] * gamma2 * rnd))
            # weights.append(delta_factor * acc_factor)

        total_weights = sum(weights)
        probs = [weight / total_weights for weight in weights]
        norm_dict = dict(zip(self.neighborhood, probs))
        sorted_dict = {n:norm_dict[n] for n in sorted(norm_dict, key=norm_dict.get, reverse=False)} # First sort Ascending order as exp function in ascending order

        if wt == 'exp':
            # Exponential Weighting
            x = 0.1 + np.exp(4 * np.linspace(0,1, len(self.neighborhood)))             # #Exponential Probabilities
            wt_dict = {node:wts*x[i] for i,(node, wts) in enumerate(sorted_dict.items())}
        
        elif wt == 'normal':
            #Normal Probabilities
            x = stats.norm.pdf(np.linspace(0, 1, len(self.neighborhood)), 0.75, 0.5) # Generate weights as per Normal Distribution
            wt_dict = {node:wts*x[i] for i,(node, wts) in enumerate(sorted_dict.items())}

        elif wt == 'skew':
            # Right skewed dist controlled by a
            x = stats.skewnorm.pdf(np.linspace(0, 1, len(self.neighborhood)), 3, loc = 0.5) # Right skewed dist
            wt_dict = {node:wts*x[i] for i,(node, wts) in enumerate(sorted_dict.items())}
        
        elif wt == 'lin':
            # Linear pdf
            x  = 0.1 + np.linspace(0, 1, len(self.neighborhood))
            wt_dict = {node:wts*x[i] for i,(node, wts) in enumerate(sorted_dict.items())}

        normwt_dict = {node:prob/np.sum(list(wt_dict.values())) for (node, prob) in wt_dict.items()}
        # sorted_normdict = sorted(norm_dict, key=normwt_dict.get, reverse=False) # Get elements from the normdict
        
        # print(sorted(norm_dict, key=norm_dict.get, reverse=True)[:], '->', sorted(acc_dict, key=acc_dict.get, reverse=True)[:])
        return normwt_dict # Return dict of nodes in descending order of weights

    def linprog_aggregation(self, nodeset, alpha, beta, gamma, zeta, eps , agg_prop):
        acc_dict, _ = self.nhbrhood_dict(nodeset, key = 'acc')
        loss_dict, _ = self.nhbrhood_dict(nodeset, key = 'loss')
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        pass

    
    def stalemodel_aggregate(self, nodeset, staleglobal, scale, agg_prop, rnd, cos_thresh):
        self.global_coscheck(staleglobal, nodeset)
        agg_full = []
        agg_conv = []
        target_nodes = int(np.floor(agg_prop * len(self.neighborhood)))
        nodelist = random.sample(self.neighborhood, target_nodes)
        # if rnd > 6:
        #     for nhbr in nodelist:
        #         if self.global_cosvals[nhbr][-1] >= cos_thresh:
        #             agg_full.append(nhbr)
        #         else:
        #             agg_conv.append(nhbr)
        #     print(f' Node: {self.idx} Agg_full {agg_full}, Agg Conv {agg_conv} Agg Vals {self.global_cosvals}', flush = True, end = ',')
        # else:
        #     agg_full = self.neighborhood[:]
        agg_full = sorted(self.global_cosvals, key=self.cos_vals.get, reverse=True)[:target_nodes]
        agg_full.append(self.idx)    

        agg_model = selective_aggregate(nodeset, agg_full, scale, agg_conv = agg_conv, noise=False)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()

    def gradient_ranking(self, nodeset, global_refmodel, scale: dict, cos_lim, global_ref = False):
        #Reset ranked Neighborhood value to allow access to entire Neighborhood in every round
        self.ranked_nhood = self.neighborhood.copy()
        self_cosdict = {i:None for i in self.neighborhood}
        global_cosdict = {i:None for i in self.neighborhood}
        global_model = copy.deepcopy(global_refmodel)
        nhbrgrad_dict = {i:None for i in self.neighborhood}
        
        self.grads = test_gradients(self.model, self.grad_testloader)
        global_grads = test_gradients(global_model.cuda(), self.testloader)

        self.grad_mag = np.linalg.norm(self.grads)
        globalgrad_mag = np.linalg.norm(global_grads)

        # print(f'Node - {self.idx}--> Nhood Before updates {self.ranked_nhood} vs Actual Nhood {self.neighborhood}')
        
        for nhbr in self.neighborhood:
            self.neighbor_grads[nhbr] = test_gradients(nodeset[nhbr].model, self.testloader)
            
            # Gradient CoSine Sim ranking
            self_cosdict[nhbr] = (self.grads @ self.neighbor_grads[nhbr].T)/(np.linalg.norm(self.grads) * np.linalg.norm(self.neighbor_grads[nhbr]))
            global_cosdict[nhbr] = (global_grads @ self.neighbor_grads[nhbr].T)/(np.linalg.norm(global_grads) * np.linalg.norm(self.neighbor_grads[nhbr]))

            # Gradient Magnitude ranking
            nhbrgrad_dict[nhbr] = np.linalg.norm(self.neighbor_grads[nhbr])
            
        if global_ref == False:
            # self.ranked_nhood = [k for k, v in self_cosdict.items() if v >= cos_lim] # Cosine Sim based
            self.ranked_nhood = [nhbr for nhbr in self.ranked_nhood if nhbrgrad_dict[nhbr] >= self.grad_mag] # Mag based
            # agg_full = [nhbr for nhbr in self.ranked_nhood if nhbrgrad_dict[nhbr] >= self.grad_mag] # Adding to full model aggregation
        else:
            # self.ranked_nhood = [k for k, v in global_cosdict.items() if v >= cos_lim] # Cosine Sim based
            self.ranked_nhood = [nhbr for nhbr in self.ranked_nhood if nhbrgrad_dict[nhbr] >= globalgrad_mag] # Mag based
            # agg_full = [nhbr for nhbr in self.ranked_nhood if nhbrgrad_dict[nhbr] >= globalgrad_mag] # Adding to Full Model Aggregation

        # print(f'Node-{self.idx}-> Ranked {self.ranked_nhood} and Original {self.neighborhood}')
        
        # Add remaining to agg_conv
        # agg_conv = [nhbr for nhbr in self.neighborhood if nhbr not in agg_full]
        # If all neighbors being removed, remove only the smallest one
        if len(self.ranked_nhood) == 0: 
            self.ranked_nhood = self.neighborhood.copy()
            selfref_remove = min(self_cosdict, key = self_cosdict.get) # Remove nhbr with min cos sim
            globalref_remove = min(global_cosdict, key = global_cosdict.get) # Remove nhbr with min cos sim wr.t. global
            gradref_remove = min(nhbrgrad_dict, key = nhbrgrad_dict.get) # Remove lowest grad magnitude
            self.ranked_nhood.remove(gradref_remove)    

        print(f'Nhbr {self.neighborhood} Ranked {self.ranked_nhood}')
        
            # if global_ref == False:
            #     self.ranked_nhood.remove(selfref_remove)
                
            # else:
            #     self.ranked_nhood.remove(globalref_remove)

        # Changed: Modding weight of most similar nodes                
        # sim_nodes = self.ranked_nhood.copy()
        # self.ranked_nhood = self.neighborhood.copy()

        # for node in sim_nodes:
        #     scale[node] += 0.01

        # print(f'Final Updated Nhood{self.ranked_nhood} vs Actual Nhood {self.neighborhood}', '\n')

        del global_model
        gc.collect()
        
class Servers:
    def __init__(self, server_id, model, records = False):
        self.idx = server_id
        self.model = copy.deepcopy(model)
        if records == True:
            self.avgtrgloss = []
            self.avgtrgacc = []
            self.avgtestloss = []
            self.avgtestacc =[]
             
    def aggregate_servers(self, server_set, nodeset):
        scale = {server_id:1.0 for server_id in range(len(server_set)) }
        global_model = aggregate(server_set, list(range(len(server_set))), scale)
        self.model.load_state_dict(global_model.state_dict())

        for server in server_set:
            server.model.load_state_dict(self.model.state_dict())
            
        for node in nodeset:
            node.model.load_state_dict(self.model.state_dict())
        del global_model
        gc.collect()
        
    def aggregate_clusters(self, nodeset, assigned_nodes, scale, prop):
        nodelist = random.sample(assigned_nodes, int(prop*len(assigned_nodes)))
        server_agg_model = aggregate(nodeset, nodelist, scale)
        self.model.load_state_dict(server_agg_model.state_dict())
        del server_agg_model
        gc.collect()