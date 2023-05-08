import copy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, SubsetRandomSampler, Subset

from DNN import *

def dataset_approve(dataset:'str'):
    if dataset == 'mnist': # Num labels will depend on the class in question
        location = '../data/'
        num_labels = 10
    elif dataset == 'cifar':
        location = '../data/'
        num_labels = 10
    elif dataset == 'fashion':
        location = '../data/'
        num_labels = 10
    return location, num_labels


class DataSubset(Dataset):
    """
    Takes the dataset, distribution list and node as arguments.
    """
    
    def __init__(self, dataset, datadist, node=None):
        self.dataset = dataset
        self.datadist = datadist
        if node == None: # Added to cater for a common val set across all nodes
            self.indx = list(self.datadist)
        else:
            self.indx = list(self.datadist[node])       
    
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.indx[item]]
#         image = self.dataset.data[item]
#         label = self.dataset.targets[item]
        return torch.tensor(image), torch.tensor(label)

def dataset_select(dataset, location, in_ch, val_size = 0.1):
    """ 
    Select from MNIST, CIFAR-10 or FASHION-MNIST
    """
    ## MNIST
    if dataset == 'mnist' or dataset == 'fashion':
        if in_ch == 1:
           ### Choose transforms
            transform = transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))])
        elif in_ch == 3:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize(224), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            transforms.Grayscale(num_output_channels = 3)]) #transforms.Lambda(lambda x: x.expand(3, -1, -1))

        if dataset == 'mnist':
            traindata = torchvision.datasets.MNIST(root = location, train = True, download = True, transform = transform)
            testdata = torchvision.datasets.MNIST(root = location, train = False, download = True, transform = transform)


        elif dataset == 'fashion':
            traindata = datasets.FashionMNIST(root = location, download = True, train = True, transform = transform)
            testdata = datasets.FashionMNIST(root =location, download = True, train = False, transform = transform)

    ## CIFAR
    elif dataset == 'cifar':
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #         transforms.Resize(224),

        traindata = torchvision.datasets.CIFAR10(root= location, train = True, download = True, transform = transform)
        testdata = torchvision.datasets.CIFAR10(root = location, train = False, download = True, transform = transform)


    # # Creating validation dataset
    # num_samples = len(testdata)
    # num_vals = int(num_samples * val_size)
    # val_sampler = SubsetRandomSampler(range(num_vals))
    # valset = Subset(testdata, val_sampler)
    # # Sampler for remaining testdata
    # test_sampler = SubsetRandomSampler(range(num_vals, num_samples))
    # testset = Subset(testdata, test_sampler)


    # Create validation set
    test_idx, val_idx = train_test_split(np.arange(len(testdata)), test_size = val_size, random_state = 999, shuffle = True, stratify = testdata.targets)

    return traindata, testdata, test_idx, val_idx

def dict_creator(modes, dataset, num_labels, in_channels, num_nodes, num_rounds, wt_init):
        
    # Same weight initialization
    if wt_init == 'same':       
        same_wt_basemodel = Net(num_labels, in_channels, dataset)
        model_dict = {i:copy.deepcopy(same_wt_basemodel).cuda() for i in range(num_nodes)}
    elif wt_init == 'diff':
        model_dict = {i:Net(num_labels, in_channels, dataset).cuda() for i in range(num_nodes)}   
    
    recorder = {node:[] for node in range(num_nodes)}
    ## Model Dictionary for each of the Fed Learning Modes
    ## Model Dictionary Initialization

    mode_model_dict = {key:None for key in modes}
    mode_trgloss_dict = {key:None for key in modes}
    mode_testloss_dict = {key:None for key in modes}
    mode_avgloss_dict = {key:[] for key in modes}
    mode_acc_dict = {key:None for key in modes}
    mode_avgacc_dict = {key:[] for key in modes}
    
    basemodel_keys = model_dict[0].state_dict().keys()
    layer_dict = {layer:[] for layer in basemodel_keys}
    nodelayer_dict = {node:copy.deepcopy(layer_dict) for node in range(num_nodes)}   
    divergence_dict = {mode:copy.deepcopy(nodelayer_dict) for mode in modes if mode != 'SGD'}  
    
    # Create separate copies for each mode
    for mode in modes:
        if mode != 'SGD':
            mode_model_dict[mode] = copy.deepcopy(model_dict)
            mode_trgloss_dict[mode] = copy.deepcopy(recorder)
            mode_testloss_dict[mode] = copy.deepcopy(recorder)
            mode_acc_dict[mode] = copy.deepcopy(recorder)
        elif mode == 'SGD':
            mode_model_dict[mode] = copy.deepcopy(same_wt_basemodel).cuda()
            mode_trgloss_dict[mode] = []
            mode_testloss_dict[mode] = []
            mode_acc_dict[mode] = []
    return mode_model_dict, mode_trgloss_dict, mode_testloss_dict, mode_avgloss_dict, mode_acc_dict, mode_avgacc_dict, divergence_dict

        
