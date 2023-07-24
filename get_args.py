import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type = int, default = 8, help = 'Batch size for the dataset')
    parser.add_argument('-t', type = int, default = 8, help = 'Batch size for the test dataset')
    parser.add_argument('-d', type = str, default = 'mnist', help='Name of the dataset : mnist, cifar10 or fashion')
    parser.add_argument('-n', type = int, default = 40, help='Number of nodes')
    parser.add_argument('-c', type = int, default = 9, help='Number of clusters')
    parser.add_argument('-ser', type = int, default = 3, help='Number of servers')
    parser.add_argument('-emin', type = int, default = 1, help='Min Number of epochs')
    parser.add_argument('-emax', type = int, default = 6, help='Max Number of epochs')
    parser.add_argument('-sim', type = float, default = 0.9996, help='Cosine Similarity Threshold')
    parser.add_argument('-r', type = int, default = 30, help='Number of federation rounds')
    parser.add_argument('-D', type = int, default = 5, help='Delay induced for Stale Global Models')
    parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
    parser.add_argument('-s', type = int, default = 0, help = ' Skew for Exteme Non-IID distribution. Min: 1 - Max: Num of classes') # Positive skew overrides alpha value
    parser.add_argument('-a', type = float, default = 0.1, help = 'Alpha parameter (+ve only) for Dirichlet based non-IID distribution')
    parser.add_argument('-prop', type = float, default = 0.9, help = 'Proportion of nodes chosen for server aggregation : 0.0-1.0')
    parser.add_argument('-aggprop', type = float, default = 0.9, help = 'Aggregation-Proportion: Proportion of nodes in neighborhood for device-level aggregation: 0.0-1.0')
    parser.add_argument('-model', type = str, default = 'shallow', help = 'Base model type to run the experiments')
    parser.add_argument('-clint', type=float, default = 0.15, help='Probability of Edges within a cluster')
    parser.add_argument('-clext', type=float, default = 0.2, help='Probability of Edges with nodes external to Cluster of current membership')    
    
    args = parser.parse_args()
    return args