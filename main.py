import sys
from predict import predict
from train import train

#define constants
description=''
base_dir=''
model_dir=''
train_dir=''
test_dir=''
MODEL_NAME="TRANSFER_LEARNING_VGG16"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS=int(0)

def main():
    if len(sys.argv) != 4:
        print ("""Arguements error \n
            Usage:  python3 segm_unet.py dataset_path number_of_epochs comment \n""")
        sys.exit(0)
    
    description=sys.argv[3]
    base_dir=sys.argv[1]
    model_dir=base_dir+"model"
    train_dir=base_dir+"train"
    test_dir=base_dir+"test"
    EPOCHS=int(sys.argv[2])

    train()
    predict()




if __name__ == "__main__":
    main()
