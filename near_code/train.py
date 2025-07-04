import json
import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# import program_learning
from algorithms import ASTAR_NEAR, IDDFS_NEAR, MC_SAMPLING, MCTS, ENUMERATION, GENETIC, RNN_BASELINE
from dsl_current import DSL_DICT, CUSTOM_EDGE_COSTS
from eval import test_set_eval
from program_graph import ProgramGraph
from utils.data import prepare_datasets
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program, print_program_dict, log_and_print


def parse_args():
    parser = argparse.ArgumentParser()
    # Args for experiment setup
    parser.add_argument('-t', '--trial', type=int, required=True,
                        help="trial ID")
    parser.add_argument('--exp_name', type=str, required=True,
                        help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # Args for data
    parser.add_argument('--train_data', type=str, required=True,
                        help="path to train data")
    parser.add_argument('--test_data', type=str, required=True, 
                        help="path to test data")
    parser.add_argument('--valid_data', type=str, required=False, default=None,
                        help="path to val data. if this is not provided, we sample val from train.")    
    parser.add_argument('--train_labels', type=str, required=True,
                        help="path to train labels")
    parser.add_argument('--test_labels', type=str, required=True, 
                        help="path to test labels")
    parser.add_argument('--valid_labels', type=str, required=False, default=None,
                        help="path to val labels. if this is not provided, we sample val from train.")     
    parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
                        help="input type of data")
    parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
                        help="output type of data")
    parser.add_argument('--input_size', type=int, required=True,
                        help="dimenion of features of each frame")
    parser.add_argument('--output_size', type=int, required=True, 
                        help="dimension of output of each frame (usually equal to num_labels")
    parser.add_argument('--num_labels', type=int, required=True, 
                        help="number of class labels")

    # Args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.01,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")

    # Args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=0.8,
                        help="split training set for validation."+
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50, 
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=6,
                        help="training epochs for symbolic programs")
    parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits"],
                        help="loss function for training")
    parser.add_argument('--class_weights', type=str, required=False, default = None,
                        help="weights for each class in the loss function, comma separated floats")

    # Args for algorithms
    parser.add_argument('--algorithm', type=str, required=True, 
                        choices=["mc-sampling", "mcts", "enumeration", "genetic", "astar-near", "iddfs-near", "rnn"],
                        help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR"+
                        " (>1.0 prunes aggressively in this case)")    
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    full_exp_name = "{}_{}_{:03d}".format(args.exp_name, args.algorithm, args.trial)

    save_path = os.path.join(args.save_dir, full_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_data = np.load(args.train_data)
    test_data = np.load(args.test_data)
    valid_data = None
    train_labels = np.load(args.train_labels)
    test_labels = np.load(args.test_labels)
    valid_labels = None
    assert train_data.shape[-1] == test_data.shape[-1] == args.input_size

    if args.valid_data is not None and args.valid_labels is not None:
        valid_data = np.load(args.valid_data)
        valid_labels = np.load(args.valid_labels)
        assert valid_data.shape[-1] == args.input_size

    batched_trainset, validset, testset = prepare_datasets(train_data, valid_data, test_data, train_labels, valid_labels, 
        test_labels, normalize=args.normalize, train_valid_split=args.train_valid_split, batch_size=args.batch_size)

    # TODO allow user to choose device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    if args.class_weights is None:
        if args.lossfxn == "crossentropy":
            lossfxn = nn.CrossEntropyLoss()
        elif args.lossfxn == "bcelogits":
            lossfxn = nn.BCEWithLogitsLoss()
    else:
        class_weights = torch.tensor([float(w) for w in args.class_weights.split(',')])
        if args.lossfxn == "crossentropy":
            lossfxn = nn.CrossEntropyLoss(weight = class_weights)
        elif args.lossfxn == "bcelogits":
            lossfxn = nn.BCEWithLogitsLoss(weight = class_weights)


    if device != 'cpu':
        lossfxn = lossfxn.cuda()

    train_config = {
        'lr' : args.learning_rate,
        'neural_epochs' : args.neural_epochs,
        'symbolic_epochs' : args.symbolic_epochs,
        'optimizer' : optim.Adam,
        'lossfxn' : lossfxn,
        'evalfxn' : label_correctness,
        'num_labels' : args.num_labels
    }

    # Initialize logging
    init_logging(save_path)
    log_and_print("Starting experiment {}\n".format(full_exp_name))

    # Initialize program graph
    program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS, args.input_type, args.output_type, args.input_size, args.output_size,
        args.max_num_units, args.min_num_units, args.max_num_children, args.max_depth, args.penalty, ite_beta=args.ite_beta)

    # Initialize algorithm
    if args.algorithm == "astar-near":
        algorithm = ASTAR_NEAR(frontier_capacity=args.frontier_capacity)
    elif args.algorithm == "iddfs-near":
        algorithm = IDDFS_NEAR(frontier_capacity=args.frontier_capacity, initial_depth=args.initial_depth, 
            performance_multiplier=args.performance_multiplier, depth_bias=args.depth_bias, exponent_bias = args.exponent_bias)
    elif args.algorithm == "mc-sampling":
        algorithm = MC_SAMPLING(num_mc_samples=args.num_mc_samples)
    elif args.algorithm == "mcts":
        algorithm = MCTS(num_mc_samples=args.num_mc_samples)
    elif args.algorithm == "enumeration":
        algorithm = ENUMERATION(max_num_programs=args.max_num_programs)
    elif args.algorithm == "genetic":
        algorithm = GENETIC(population_size=args.population_size, selection_size=args.selection_size, num_gens=args.num_gens,
            total_eval=args.total_eval, mutation_prob=args.mutation_prob, max_enum_depth=args.max_enum_depth)
    elif args.algorithm == "rnn":
        algorithm = RNN_BASELINE()
    else:
        raise NotImplementedError

    # Run program learning algorithm
    best_programs = algorithm.run(program_graph, batched_trainset, validset, train_config, device)
    best_program_time = 0.0

    if args.algorithm == "rnn":
        # special case for RNN baseline
        best_program = best_programs
    else:
        # Print all best programs found
        log_and_print("\n")
        log_and_print("BEST programs found:")
        for item in best_programs:
            print_program_dict(item)
        best_program = best_programs[-1]["program"]
        best_program_time = best_programs[-1]["time"]

    # Save best program
    pickle.dump(best_program, open(os.path.join(save_path, "program.p"), "wb"))

    # Evaluate best program on test set
    true_vals, predicted_vals, report = test_set_eval(best_program, testset, args.output_type, args.output_size, args.num_labels, device)
    with open(os.path.join(save_path, "test_results.json"), "w") as f:
        json.dump(
            dict(
                true_vals=true_vals.cpu().numpy().tolist(),
                predicted_vals=predicted_vals.cpu().numpy().tolist(),
                report=report,
                time=best_program_time,
                program=print_program(best_program, ignore_constants=False),
                program_str=print_program(best_program, ignore_constants=True),
            ),
            f,
        )
    log_and_print("ALGORITHM END \n\n")
    