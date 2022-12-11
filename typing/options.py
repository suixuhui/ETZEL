import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./models')
parser.add_argument('-DATA_DIR', type=str, default='../data')
parser.add_argument('-INSTANCES_DIR', type=str, default='../data/instances')

# model
parser.add_argument("-model", type=str, default='bert')
parser.add_argument("-max_seq_length", type=int, default=128)
parser.add_argument("-test_model", type=str, default=None)
parser.add_argument("-load_model", type=str, default=None) # "./models/berttype"

# training
parser.add_argument("-n_epochs", type=int, default=300)
parser.add_argument("-dir_name", type=str, default="type")
parser.add_argument("-mode", type=str, default="train")
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=1e-5)
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-gpu", type=int, default=0, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("-tune_wordemb", action="store_const", const=True, default=True)
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')

parser.add_argument('-bert_dir', type=str, default="bert-base-uncased")

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
