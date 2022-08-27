#!python3
import sys
from params import weights_path, program
from predict import predict_func
from train import train, load_model
def train_and_predict():
    train()
    predict_func()

def predict():
    load_model(weights_path)
    predict_func()

if __name__ == "__main__":
    if program=='ONLY_PREDICT':
        predict()
    if program=='TRAIN_AND_PREDICT':
        train_and_predict()
        
