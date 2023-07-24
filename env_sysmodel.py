import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
# import heapq
# import pickle
import sys, gc

import torch
# import torchvision
import torchvision.models as models
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from DNN import *
from data_utils import DataSubset
from devices import Nodes, Servers
from utils import constrained_sum

class system_model:
    """
    Creates the requisite System model / environment.
    Generates clusters, defines neighborhood and graph.
    Creates dictionary for records.
    """    
    def __init__(self, num_nodes, num_clusters, num_servers, prob_int = 0.95, prob_ext = 0.02):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.num_servers  = num_servers
        self.max_size = int((self.num_nodes / self.num_clusters) + 3) 
        self.min_size = 4
        self.prob_int = prob_int #Probability of intra-parition edges
        self.prob_ext = prob_ext #Probability of inter-partition edges
        
        self.generate_clusters() # Generate Network Layout
        self.create_graph() # Generate graph
        self.map_neighborhood() # Map node neighborhood
        self.graph_stats() # Graph statistics
        self.create_servers() # Create hierarchical servers.
        self.cluster_head_select() #Select Cluster heads. Applicable should only be flag based
        self.draw_graph()
    
    def generate_clusters(self):
        self.cluster_sizes = []
        while sum(self.cluster_sizes) != self.num_nodes:
            self.cluster_sizes = np.random.randint(self.min_size, self.max_size, self.num_clusters)    
        self.cluster_set = []
        for i, _ in enumerate(self.cluster_sizes):
            temp = list(range(sum(self.cluster_sizes[:i]), sum(self.cluster_sizes[:i+1])))
            self.cluster_set.append(temp)
        self.avg_cluster_size = int(sum(self.cluster_sizes) / self.num_clusters)
        print(f'The generated cluster set is {self.cluster_set}')
        
    def create_graph(self):
        self.graph = nx.random_partition_graph(self.cluster_sizes, self.prob_int, self.prob_ext)
        self.graph_density =  nx.density(self.graph)
    
    def draw_graph(self):
        plt.figure(figsize = (20,12))
        nx.draw(self.graph, with_labels = True, font_weight = 'bold')
        plt.title('Network graph', fontsize = 28)

        plt.figtext(0.75, 0.95, 'Nodes:{}, Clusters:{}, Servers:{}'.format(self.num_nodes, self.num_clusters, self.num_servers) ,  fontsize = 20)
        plt.figtext(0.75, 0.9, 'Avg Cluster Size:{}'.format(self.avg_cluster_size) ,  fontsize = 20)
        plt.figtext(0.75, 0.85, 'Avg Degree:{}'.format(self.avg_neighbors) ,  fontsize = 20)
        plt.figtext(0.75, 0.8, 'Graph Density : {:.2f}'.format(self.graph_density), fontsize = 20)
        plt.savefig('Network_graph')
            
    def map_neighborhood(self):
        neighborhood = []
        for node in range(self.num_nodes):
            temp = [neighbor for neighbor in self.graph.neighbors(node)]
            neighborhood.append(temp)
        self.neighborhood_map = neighborhood

        self.avg_neighbors = 0
        for i in range(self.num_nodes):
            self.avg_neighbors += len(self.neighborhood_map[i])
        self.avg_neighbors /= self.num_nodes
        self.avg_neighbors = int(self.avg_neighbors)


    def graph_stats(self, plot_fig = False):
        self.Lp = nx.normalized_laplacian_matrix(self.graph)
        self.ev = np.linalg.eigvals(self.Lp.A)
#         print("Largest eigenvalue:", max(self.ev))
#         print("Smallest eigenvalue:", min(self.ev))
        if plot_fig == True:
            plt.figure()
            plt.hist(self.ev, bins=100)  # histogram with 100 bins
            plt.xlim(0, 5)  # eigenvalues between 0 and 2
            plt.savefig('Eigen Value: Histogram')
    
    def create_servers(self):
        # Length of cluster_ids is always going to be equal to num_servers.
        # Randomly assign a number of clusters to each server. 
        target_indx = constrained_sum(self.num_servers, self.num_clusters)
        # Choose cluster_ids based on the assigned number of clusters
        # e.g Server-1 assigned 2 clusters may be assigned any 2 cluster ids
        self.server_groups = []
        cluster_list = list(range(self.num_clusters))
        # Randomize the cluster list
        random.shuffle(cluster_list)
        for i in target_indx:
            temp = cluster_list[:i]
            cluster_list = [item for item in cluster_list if item not in temp]
            self.server_groups.append(temp)
        # List of cluster_ids == randomized clusters assigned to each server in sequential order.
        print(f'With {self.server_groups} servers and server_targets have been configured')
        
        # Assign nodes to server for better trackng
        self.server_nodes = {i:[] for i in range(self.num_servers)}
        for i in range(self.num_servers):
            for j in range(len(self.server_groups[i])):
                self.server_nodes[i] += self.cluster_set[self.server_groups[i][j]]
                
    def cluster_head_select(self):
        self.cluster_heads = []
        for cluster in self.cluster_set:
            temp = (random.sample(cluster, 1))[0] # random.sample returns a list thus subscripting allows extracting the item
            self.cluster_heads.append(temp)
            
class FL_Modes(Nodes):
    modes_list = []
    #'d2d', dataset, num_epochs, num_nodes, base_model, num_labels, in_ch, traindata, train_dist, testdata, test_dist, dataset, batch_size, env.neighborhood_map, env.Lp
    def __init__(self, name, dataset, num_epochs, num_rounds, num_nodes, base_model, 
                 num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, valdata_dist,
                 batch_size, nhood, env_Lp, num_clusters, **kwargs):
        # Kwargs include d2d_agg_flg, ch_agg_flg, hserver_agg_flg, 
        
        super().__init__(1, base_model, num_labels, in_channels, traindata, traindata_dist, 
                         testdata, testdata_dist, dataset, batch_size, [0,0])
        self.name = name
        self.dataset = dataset
        self.epochs = num_epochs
        self.rounds = num_rounds
        self.base_model = copy.deepcopy(base_model)
        self.batch_size = batch_size
        self.cfl_model = copy.deepcopy(self.base_model)
        self.cfl_model_dict = {}
        self.valloader = DataLoader(DataSubset(testdata, valdata_dist), batch_size = batch_size)

            
        if self.name != 'sgd':
            self.num_clusters = num_clusters
            self.num_nodes = num_nodes
            self.nodeweights(traindata, traindata_dist, scale = 'one')
            self.form_nodeset(num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, nhood)
            
            # Aggregation Flags
            self.d2d_agg_flg = kwargs['d2d_agg_flg']
            self.ch_agg_flg = kwargs['ch_agg_flg']
            self.hserver_agg_flg = kwargs['hserver_agg_flg']
            self.inter_ch_agg_flg = kwargs['inter_ch_agg_flg']
                        
        self.cluster_trgloss = {cluster_id:[] for cluster_id in range(num_clusters)}
        self.cluster_trgacc = {cluster_id:[] for cluster_id in range(num_clusters)}
        self.cluster_testloss = {cluster_id:[] for cluster_id in range(num_clusters)}
        self.cluster_testacc = {cluster_id:[] for cluster_id in range(num_clusters)}
        self.cluster_valacc = {cluster_id:[] for cluster_id in range(num_clusters)}
        self.cluster_valloss = {cluster_id:[] for cluster_id in range(num_clusters)}
        
        # Mode records
        self.avgtrgloss = []
        self.avgtestloss = []
        self.avgtrgacc = []
        self.avgtestacc = []
        self.avgvalloss = []
        self.avgvalacc = []
        
        FL_Modes.modes_list.append(self.name)      

    def form_nodeset(self, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, nhood):
        # base_model, num_labels, in_channels, traindata, trg_dist, testdata, test_dist, dataset, batch_size, node_neighborhood
        self.nodeset = []
        for idx in range(self.num_nodes):
            node_n_nhood = nhood[idx]
#             node_n_nhood.append(idx)
            self.nodeset.append(Nodes(idx, self.base_model, num_labels, in_channels, traindata, traindata_dist, 
                                      testdata, testdata_dist, self.dataset, self.batch_size, node_n_nhood))
    
    def form_serverset(self, num_servers, num_labels, in_channels, dataset):
        self.serverset = []
        for idx in range(num_servers):
            self.serverset.append(Servers(num_servers, self.base_model))
        #Append 1-additioal server to act as a Global server
        self.serverset.append(Servers(num_servers, self.base_model))
        
    def graph_weights(self, Laplacian):
        """ Edge Weights for the environment graph"""
        self.edge_weights = Laplacian.toarray()
        
    def nodeweights(self, traindata, traindata_dist, scale = 'one'):  
        """ Node weights based on either the total number of network nodes, # of trg datasamples or 1s"""
        if scale == 'degree':
            self.weights ={i:1.0/self.num_nodes for i in range(self.num_nodes)}
            
        elif scale == 'proportional':
            self.weights = {i:len(traindata_dist[i])/len(traindata) for i in range(self.num_nodes)}
        
        elif scale == 'one':
            self.weights = {i:1.0 for i in range(self.num_nodes)}
            
    def update_round(self, emin, emax):
        """
        Perform local training on the entire worker set of cuda_models.
        Takes min_epochs and max_epochs and radomly selects for each node, the number of the epochs between the two.
        """
        temp_acc = []
        temp_loss = []

        for node in tqdm(self.nodeset, desc='Node Update Status'):
            try:
                epochs = np.random.randint(emin, emax) # Different local updates. Set to a number if homogenous training needed
            except:
                epochs = abs(emax)
            # print(f'Check Node {i}-model on Cuda : {next(node.model.parameters()).is_cuda}')
            node.model.to('cuda')
            node.local_update(epochs) # Asynchronous Aggregation. Change epochs to self.epochs for synchronous operation.
            temp_acc.append(node.trgacc[-1])
            temp_loss.append(node.trgloss[-1])

        self.avgtrgloss.append(sum(temp_loss)/self.num_nodes)
        self.avgtrgacc.append(sum(temp_acc)/self.num_nodes)

    def val_round(self):
        temp_acc = []
        temp_loss = []
        for node in tqdm(self.nodeset, desc='Node validation status'):
            node.node_val(self.valloader)
            temp_acc.append(node.valacc[-1])
            temp_loss.append(node.valloss[-1])    
        self.avgvalacc.append(np.round(np.mean(temp_acc), 3))
        self.avgvalloss.append(np.round(np.mean(temp_loss), 3))
        print(f'Avg Val Acc-{np.mean(temp_acc):0.3f} Variance {np.var(temp_acc):0.3f} max {max(temp_acc):0.3f} min {min(temp_acc):0.3f}')
        
    def test_round(self, cluster_set):
        temp_acc = []
        temp_loss = []
        for node in self.nodeset:
            node.node_test()
            temp_acc.append(node.testacc[-1])
            temp_loss.append(node.testloss[-1])    
        self.avgtestacc.append(np.round(np.mean(temp_acc), 3))
        self.avgtestloss.append(np.round(np.mean(temp_loss), 3))
        print(f'Avg Test Acc-{np.mean(temp_acc):0.3f} Variance {np.var(temp_acc):0.3f} max {max(temp_acc):0.3f} min {min(temp_acc):0.3f}')

        # Calculate glolbal and cluter averages
        self.global_avgs()
        self.cluster_avgs(cluster_set)

    def record_round_models(self):
        for node in self.nodeset:
            node.prevrnd_model.load_state_dict(node.prevmodel.state_dict())
            
    def gradient_ranking_round(self, ref_rnd, cos_lim, global_ref = False):
        # Updates ranked neighborhood on the basis of gradient rankings
        for node in self.nodeset:
            node.gradient_ranking(self.nodeset, self.cfl_model_dict[ref_rnd], self.weights, cos_lim, global_ref = global_ref)
                
    def nhood_aggregate_round(self, agg_prop):
        for node in self.nodeset:
            node.aggregate_nodes(self.nodeset, agg_prop, self.weights)
            
    def selective_aggregate_round(self, agg_prop, rnd, cos_lim):
        for node in self.nodeset:
            node.aggregate_selective(self.nodeset, self.weights, agg_prop, rnd, cos_lim)
            
    def random_aggregate_round(self):
        node_pairs = []
        node_list = list(range(len(self.nodeset)))
        while len(node_list) > 1:
            temp =  random.sample(node_list, 2)
            node_list = [item for item in node_list if item not in temp]
            node_pairs.append(temp)
        for node_pair in node_pairs:
            aggregate(self.nodeset, node_pair, self.weights)
    
    def layerwise_aggregate_round(self, agg_prop, cos_lim):
        for node in self.nodeset:
            node.aggregate_layerwise(self.nodeset, self.weights, agg_prop, cos_lim)
    
    def stalelayer_aggregate_round(self, ref_rnd, agg_prop, cos_lim):
        for node in self.nodeset:
            #    def stale_aggregate_layerwise(self, nodeset, staleglobal, scale):
            node.stale_aggregate_layerwise(self.nodeset, self.cfl_model_dict[ref_rnd], self.weights, agg_prop, cos_lim)

    def stalemodel_aggregate_round(self, ref_rnd, agg_prop, rnd, cos_lim):
        for node in self.nodeset:
            node.stalemodel_aggregate(self.nodeset, self.cfl_model_dict[ref_rnd], self.weights, agg_prop, rnd, cos_lim)

    def d2daccwt_aggregate_round(self, agg_prop, rnd, alpha, beta, gamma, eps, wt):
        for node in self.nodeset:
            node.d2daccwt_aggregate(self.nodeset, rnd, alpha, beta, gamma, eps, wt, self.weights, agg_prop)
    
    def d2dval_aggregate_round(self, rnd:int, agg_prop:float, eps, metric = 'valacc'):
        for node in self.nodeset:
            node.d2dval_aggregate(self.nodeset, rnd, eps, agg_prop, self.weights, metric)

    def d2dmem_aggregate_round(self, agg_prop:float):
        for node in self.nodeset:
            node.aggregate_nodesmemory(self.nodeset, agg_prop, self.weights)

    def d2dfullmem_aggregate_round(self, agg_prop):
        for node in self.nodeset:
            node.aggregate_nodesfullmem(self.nodeset, agg_prop, self.weights)

    def d2dmemextnd_aggregate_round(self, agg_prop:float, rnd):
        for node in self.nodeset:
            node.aggregate_memextend(self.nodeset, agg_prop, rnd, self.weights)

    def d2dmemextndrly_aggregate_round(self, agg_prop:float, rnd:int):
        for node in self.nodeset:
            node.aggregate_memextendrly(self.nodeset, agg_prop, rnd, self.weights)

    def d2dexnhbr_aggregate_round(self, agg_prop, rnd):
        for node in self.nodeset:
            node.aggregate_extndnhbr(self.nodeset, agg_prop, rnd, self.weights)

    def d2dexnhbrred_aggregate_round(self, agg_prop, rnd):
        for node in self.nodeset:
            node.aggregate_extndnhbrred(self.nodeset, agg_prop, rnd, self.weights)

    def d2dextnrly_aggregate_round(self, agg_prop, rnd):
        for node in self.nodeset:
            node.aggregate_extnrly(self.nodeset, agg_prop, rnd, self.weights)
            
    def clshead_aggregate_round(self, cluster_head, cluster_set, agg_prop):
        self.nodeset[cluster_head].aggregate_nodes(self.nodeset, agg_prop, self.weights, cluster_set = cluster_set)
        # Load CH model on all cluster nodes 
        for node in cluster_set:
            self.nodeset[node].model = copy.deepcopy(self.nodeset[cluster_head].model)
#             self.nodeset[node].model.load_state_dict(self.nodeset[cluster_head].model.state_dict())
    
    def inter_ch_aggregate_round(self, cluster_heads_list):
        random.shuffle(cluster_heads_list)
        cluster_heads = copy.deepcopy(cluster_heads_list) # The new list will be changed, hence a copy of the original
        ch_pairs = []
        if len(cluster_heads) % 2 != 0:
            while len(cluster_heads) > 1:
                temp = random.sample(cluster_heads, 2)
                cluster_heads = [ch for ch in cluster_heads if ch not in temp]
                ch_pairs.append(temp)
        else:
            while len(cluster_heads) != 0:
                temp = random.sample(cluster_heads, 2)
                cluster_heads = [ch for ch in cluster_heads if ch not in temp]
                ch_pairs.append(temp)
                
        for ch_pair in ch_pairs:
            agg_model = aggregate(self.nodeset, ch_pair, self.weights)
            for node in ch_pair:
                self.nodeset[node].model.load_state_dict(agg_model.state_dict()) 
        del agg_model
        gc.collect()
    
    def cfl_aggregate_round(self, rnd, prop, flag):
        nodelist = list(range(self.num_nodes))
        agg_count = int(np.floor(prop * len(nodelist)))
        if agg_count < 1:
            agg_count = 1
            
        sel_nodes = random.sample(nodelist, agg_count) # Random Sampling
        agg_model = aggregate(self.nodeset, sel_nodes, self.weights)
        self.cfl_model.load_state_dict(agg_model.state_dict())
        self.cfl_model_dict[rnd] = self.cfl_model
        if flag == 'CServer':
            for node in self.nodeset:
                node.model.load_state_dict(self.cfl_model.state_dict())
        
        del agg_model
        gc.collect()
       
    def server_aggregate(self, cluster_id, cluster_set):
        ref_dict = self.nodeset[0].model.state_dict()
        for cluster in cluster_id:
            for layer in ref_dict.keys():
                ref_dict[layer] = torch.stack([self.nodeset[node].model.state_dict()[layer].float() for node in cluster_set[cluster]], 0).mean(0)
    
    def global_avgs(self):
        temp_trgloss = []
        temp_trgacc = []
        temp_testacc = []
        temp_testloss = []
        temp_valloss  = []
        temp_valacc = []

        for node in self.nodeset:
            temp_trgacc.append(node.trgacc[-1])
            temp_trgloss.append(node.trgloss[-1])
            temp_testloss.append(node.testloss[-1])
            temp_testacc.append(node.testacc[-1])
            temp_valloss.append(node.valloss[-1])
            temp_valacc.append(node.valacc[-1])
        
        self.trgacc.append(sum(temp_trgacc)/self.num_nodes)   
        self.trgloss.append(sum(temp_trgloss)/self.num_nodes)
        self.testloss.append(sum(temp_testloss)/self.num_nodes)
        self.testacc.append(sum(temp_testacc)/self.num_nodes)
        self.valacc.append(sum(temp_valacc)/self.num_nodes)
        self.valloss.append(sum(temp_valloss)/self.num_nodes)
        
    def cluster_avgs(self, cluster_set):
        for cluster_id, cluster_nodes in enumerate(cluster_set):
            temp_trgloss = []
            temp_trgacc = []
            temp_testacc = []
            temp_testloss = []
            temp_valacc = []
            temp_valloss = []
            for node in cluster_nodes:
                temp_trgacc.append(self.nodeset[node].trgacc[-1])
                temp_trgloss.append(self.nodeset[node].trgloss[-1])
                temp_testloss.append(self.nodeset[node].testloss[-1])
                temp_testacc.append(self.nodeset[node].testacc[-1])
                temp_valloss.append(self.nodeset[node].valloss[-1])
                temp_valacc.append(self.nodeset[node].valacc[-1])
                
            self.cluster_trgacc[cluster_id].append(sum(temp_trgacc)/len(cluster_set[cluster_id]))
            self.cluster_trgloss[cluster_id].append(sum(temp_trgloss)/len(cluster_set[cluster_id]))
            self.cluster_testloss[cluster_id].append(sum(temp_testloss)/len(cluster_set[cluster_id]))
            self.cluster_testacc[cluster_id].append(sum(temp_testacc)/len(cluster_set[cluster_id]))
            self.cluster_valacc[cluster_id].append(sum(temp_valacc)/len(cluster_set[cluster_id]))
            self.cluster_valloss[cluster_id].append(sum(temp_valloss)/len(cluster_set[cluster_id]))

    def ranking_round(self, rnd, mode):
        for node in self.nodeset:
            node.neighborhood_divergence(self.nodeset, self.cfl_model, div_mode = 'cfl_div', normalize = False)
            node.nhood_ranking(rnd, mode, sort_scope = 1)
            
    def nhood_ranking(self, rnd, sort_crit = 'total', sort_scope= 1 , sort_type = 'min'):
        for node in range(self.num_nodes):
            if sort_crit == 'total':
                self.apply_ranking(node, self.div_dict[node], rnd, sort_scope, sort_type)
            elif sort_crit == 'conv':
                self.apply_ranking(node, self.div_conv_dict[node], rnd, sort_scope, sort_type)
            elif sort_crit == 'fc':
                self.apply_ranking(node, self.div_fc_dict[node], rnd, sort_scope, sort_type)