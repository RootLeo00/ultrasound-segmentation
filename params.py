#define constants
import sys
import datetime
import os

import argparse
#PROG: TRAIN AND PREDICT
args=sys.argv[1:]
if args and args[0].startswith("TP"):
        parser = argparse.ArgumentParser(prog='TP', usage='%(prog)s [options]')
        parser.add_argument("-TP","--train-and-predict",
                            help="choose train and predict program", 
                            action="store_true", 
                            default=False
                    )
        parser.add_argument("-P","--only-predict",
                            help="choose only predict program", 
                            action="store_true", 
                            default=False
                    )
        parser.add_argument("-v","--version", 
                     action='version', 
                     version='%(prog)s 1.0'
                    )
        parser.add_argument("-e","--epochs", 
                            nargs='+',
                            required=False,
                            help="number of epochs to train a model",
                            type=int,
                            default=3,
                            )
        parser.add_argument("-c","--comment", 
                            nargs='?',
                            required=False,
                            help="general comment printed to log file",
                            default="test raw",
                            )
        parser.add_argument("-d","--dir-dataset", 
                            nargs='+',
                            required=True,
                            help="absolute directory path where dataset is stored",
                            )
        parser.add_argument("-m","--model-name", 
                            nargs='+',
                            required=True,
                            choices=['UNET', 'TRANSFER_LEARNING_VGG16', 'TRANSFER_LEARNING_VGG19'],
                            help="name of the model to train",
                            )
        parser.add_argument("-s","--save-path", 
                            nargs='+',
                            required=False,
                            default=parser.parse_args(args=args).dir_dataset[0],
                            help="absolute directory path where store results",
                            )
else:
        parser = argparse.ArgumentParser(prog='P', usage='%(prog)s [options]')
        parser.add_argument("-TP","--train-and-predict",
                            help="choose train and predict program", 
                            action="store_true", 
                            default=False
                    )
        parser.add_argument("-P","--only-predict",
                            help="choose only predict program", 
                            action="store_true", 
                            default=False
                    )
        parser.add_argument("-v","--version", 
                            action='version', 
                            version='%(prog)s 1.0'
                            )
        parser.add_argument("-mw", "--model-weights", 
                            nargs='+',
                            required=True,
                            type=argparse.FileType('r'),
                            help="absolute file path where weights of a model are saved",
                            )
        parser.add_argument("-c","--comment", 
                            nargs='?',
                            required=False,
                            help="general comment printed to log file",
                            default="test raw",
                            )
        parser.add_argument("-m","--model-name", 
                            nargs='+',
                            required=True,
                            choices=['UNET', 'TRANSFER_LEARNING_VGG16', 'TRANSFER_LEARNING_VGG19'],
                            help="name of the model to load",
                            )
        parser.add_argument("-d","--dir-dataset", 
                            nargs='+',
                            required=True,
                            help="absolute directory path where dataset is stored",
                            )
        parser.add_argument("-s","--save-path", 
                            nargs='+',
                            required=False,
                            default=os.path.dirname(parser.parse_args(args=args).model_weights[0].name),
                            help="absolute directory path where store results",
                            )
args = parser.parse_args(args=args)

#TODO: inizializzare meglio
description=''
base_dir=''
weights_path=''
MODEL_NAME=''
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS=0
pred_dir = args.save_path+"/predictions"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if args.only_predict:
    program='ONLY_PREDICT'
    description=args.comment[0]
    base_dir=args.dir_dataset[0]
    weights_path=args.model_weights[0].name
    MODEL_NAME=args.model_name[0]
if args.train_and_predict:
    program='TRAIN_AND_PREDICT'
    description=args.comment[0]
    base_dir=args.dir_dataset[0]
    MODEL_NAME=args.model_name[0]
    EPOCHS=args.epochs[0]
    
model_dir=base_dir+"model"
train_dir=base_dir+"train"
test_dir=base_dir+"test"

  
