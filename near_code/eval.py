import argparse
import os
import pickle
import torch
import numpy as np

from utils.data import flatten_batch, prepare_datasets
from utils.evaluation import label_correctness
from utils.logging import log_and_print, print_program
from utils.training import process_batch

from sklearn.metrics import classification_report


def test_set_eval(program, testset, output_type, output_size, num_labels, device='cpu', verbose=False):
    log_and_print("\n")
    log_and_print("Evaluating program {} on TEST SET".format(print_program(program, ignore_constants=(not verbose))))
    with torch.no_grad():
        test_input, test_output = map(list, zip(*testset))
        true_vals = torch.tensor(flatten_batch(test_output)).to(device)
        # true_vals = torch.nn.functional.one_hot(true_vals.long(), num_classes=num_labels).float().to(device)
        predicted_vals = process_batch(program, test_input, output_type, output_size, device)
        if len(predicted_vals.size()) == 3:
            predicted_vals = predicted_vals.reshape(-1, num_labels)
        # predicted_vals = predicted_vals.reshape_as(true_vals)
        metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=num_labels)

        report = classification_report(
            true_vals.cpu().numpy().tolist(),
            (torch.sigmoid(predicted_vals.cpu()) > 0.5).numpy().tolist(),
            output_dict=True
        )
    log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
    log_and_print("Additional performance parameters: {}\n".format(additional_params))
    # raw numpy true vals and predicted vals.
    return true_vals, predicted_vals, report

def parse_args():
    parser = argparse.ArgumentParser()
    # Args for experiment setup
    parser.add_argument('--program_path', type=str, required=True,
                        help="path to program")

    # Args for data
    parser.add_argument('--train_data', type=str, required=True,
                        help="path to train data")
    parser.add_argument('--test_data', type=str, required=True, 
                        help="path to test data")
    parser.add_argument('--train_labels', type=str, required=True,
                        help="path to train labels")
    parser.add_argument('--test_labels', type=str, required=True, 
                        help="path to test labels")
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
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load program
    assert os.path.isfile(args.program_path)
    program = pickle.load(open(args.program_path, "rb"))

    # Load test set
    train_data = np.load(args.train_data)
    test_data = np.load(args.test_data)
    train_labels = np.load(args.train_labels)
    test_labels = np.load(args.test_labels)
    batched_trainset, validset, testset = prepare_datasets(train_data, None, test_data, train_labels, None, test_labels, normalize=args.normalize)

    # TODO allow user to choose device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    test_set_eval(program, testset, args.output_type, args.output_size, args.num_labels, device=device, verbose=False)
