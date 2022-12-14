# define constants
import sys
import datetime
import os

import argparse


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


args = sys.argv[1:]
parser = argparse.ArgumentParser(prog='PROG', usage='%(prog)s [options]')
if args and args[0].startswith("-TP"):
    # PROG: TRAIN AND PREDICT
    parser = argparse.ArgumentParser(prog='TP', usage='%(prog)s [options]')
    parser.add_argument("-TP", "--train-and-predict",
                        help="choose train and predict program",
                        action="store_true",
                        default=False
                        )
    parser.add_argument("-P", "--only-predict",
                        help="choose only predict program",
                        action="store_true",
                        default=False
                        )
    parser.add_argument("-v", "--version",
                        action='version',
                        version='%(prog)s 1.0'
                        )
    parser.add_argument("-e", "--epochs",
                        required=False,
                        help="number of epochs to train a model",
                        type=int,
                        default=3,
                        )
    parser.add_argument("-pat", "--patience",
                        required=False,
                        help="number of patience for earlystopping callback during fit",
                        type=int,
                        default=50,
                        )
    parser.add_argument("-mon", "--monitor",
                        required=False,
                        help="Metric to monitor for earlystopping and modelcheckpoint callbacks during fit",
                        default='val_loss',
                        )
    parser.add_argument("-c", "--comment",
                        nargs='?',
                        required=False,
                        help="general comment printed to log file",
                        default="no comment",
                        )
    parser.add_argument("-d", "--dir-dataset",
                        required=True,
                        help="absolute directory path where dataset is stored",
                        )
    parser.add_argument("-m", "--model-name",
                        required=True,
                        choices=['UNET', 'TRANSFER_LEARNING_VGG16',
                                 'TRANSFER_LEARNING_VGG19'],
                        help="name of the model to train",
                        )
    parser.add_argument("-s", "--save-path",
                        required=False,
                        default=parser.parse_args(args=args).dir_dataset,
                        help="absolute directory path where store results",
                        )

elif args and args[0].startswith("-P"):
    # PROG: ONLY PREDICT
    parser = argparse.ArgumentParser(prog='P', usage='%(prog)s [options]')
    parser.add_argument("-TP", "--train-and-predict",
                        help="choose train and predict program",
                        action="store_true",
                        default=False
                        )
    parser.add_argument("-P", "--only-predict",
                        help="choose only predict program",
                        action="store_true",
                        default=False
                        )
    parser.add_argument("-v", "--version",
                        action='version',
                        version='%(prog)s 1.0'
                        )
    parser.add_argument("-mw", "--model-weights",
                        required=True,
                        type=argparse.FileType('r'),
                        help="absolute file path where weights of a model are saved",
                        )
    parser.add_argument("-c", "--comment",
                        nargs='?',
                        required=False,
                        help="general comment printed to log file",
                        default="test raw",
                        )
    parser.add_argument("-m", "--model-name",
                        required=True,
                        choices=['UNET', 'TRANSFER_LEARNING_VGG16',
                                 'TRANSFER_LEARNING_VGG19'],
                        help="name of the model to load",
                        )
    parser.add_argument("-d", "--dir-dataset",
                        required=True,
                        type=dir_path,
                        help="absolute directory path where dataset is stored",
                        )
    parser.add_argument("-s", "--save-path",
                        required=False,
                        type=dir_path,
                        default=parser.parse_args(args=args).dir_dataset,
                        help="absolute directory path where store results",
                        )
args = parser.parse_args(args=args)

#parameters initialization
description = ''
base_dir = ''
weights_path = ''
MODEL_NAME = ''
SIZE_X = 224
SIZE_Y = 224
IMAGE_SIZE = (SIZE_X, SIZE_Y)
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 0
patience = 50
monitor= 'val_loss'
pred_dir = args.save_path+"/predictions" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if args.only_predict:
    save_path, file_name=os.path.split(os.path.abspath(args.model_weights.name))
    print(save_path)
    program = 'ONLY_PREDICT'
    description = args.comment
    base_dir = args.dir_dataset
    weights_path = args.model_weights.name
    MODEL_NAME = args.model_name
    pred_dir = save_path+"/predictions" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(pred_dir)
if args.train_and_predict:
    program = 'TRAIN_AND_PREDICT'
    description = args.comment
    base_dir = args.dir_dataset
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    patience = args.patience
    monitor=args.monitor

model_dir = base_dir+"model"
train_dir = base_dir+"train"
test_dir = base_dir+"test"
