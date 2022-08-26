#!python3
import sys
from params import weights_path, program
from predict import predict_func
from train import train, load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_and_predict():
    # if len(sys.argv) != 4:
    #     print ("""Arguments error \n
    #         Usage:  python3 main.py dataset_path number_of_epochs comment \n""")
    #     sys.exit(0)
    
    train()
    predict_func()

def predict():
    # if len(sys.argv) != 3:
    #     print ("""Arguments error \n
    #     Usage:  python3 main.py weights_path comment \n""")
    # sys.exit(0)
    load_model(weights_path)
    predict_func()

if __name__ == "__main__":
    if program=='ONLY_PREDICT':
        predict()
    if program=='TRAIN_AND_PREDICT':
        train_and_predict()
        
