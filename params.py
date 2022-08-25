#define constants
import sys
import datetime

description=sys.argv[3]
base_dir=sys.argv[1]
model_dir=base_dir+"model"
train_dir=base_dir+"train"
test_dir=base_dir+"test"
MODEL_NAME="UNET"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS=int(sys.argv[2])
pred_dir = base_dir+"predictions"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")