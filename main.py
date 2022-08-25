#!python3
import sys
import params
from predict import predict_func
from train import train

def main():
    if len(sys.argv) != 4:
        print ("""Arguements error \n
            Usage:  python3 segm_unet.py dataset_path number_of_epochs comment \n""")
        sys.exit(0)
    
    train()
    predict_func()




if __name__ == "__main__":
    main()
