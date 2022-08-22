#!/usr/bin/env python3
import datetime

from main import IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, MODEL_NAME, base_dir, model_dir, train_dir, test_dir, description
from train import history_TL, history_FT

if MODEL_NAME == "TRANSFER_LEARNING_VGG16":
    import models.vgg16 as model_module
elif MODEL_NAME == "TRANSFER_LEARNING_VGG19":
    import models.vgg19 as model_module
elif MODEL_NAME == "SEGM_UNET":
    import models.unet as model_module

def print_details(pred_dir, model):
    """
    Prints details of the model.
    """

    f = open(pred_dir+"/prediction" +
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".txt", "a")
    f.write("--START--\n")
    f.write("Using dataset at: "+base_dir+"\n")
    f.write("IMAGE SIZE: "+str(IMAGE_SIZE)+"\n")
    f.write("NUMBER OF CLASSES: "+str(NUM_CLASSES)+"\n")
    f.write("BATCH SIZE: "+str(BATCH_SIZE)+"\n")
    f.write("NUMBER OF EPOCHS: "+str(EPOCHS)+"\n")

    f.write("****MY COMMENT****\n")
    f.write(description)
    f.write("\n*********\n")

    f.write("Using "+str(len(model_module.train_input_img_paths)) +
            " train images loaded from: "+model_module.train_dir+"\n")
    f.write("Using "+str(len(model_module.test_input_img_paths)) +
            " test images are loaded from: "+test_dir+"\n")
    f.write("The model used for this training is: " + MODEL_NAME+"\n")
    f.write("Model summary:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n--COMPILE AND EVALUATE MODEL--\n")
    f.write("optimizer: "+str(model_module.optimizer)+"\n")
    f.write("loss: " + str(model_module.loss) + "\n")
    f.write("with class_weights: "+str(model_module.class_weights)+"\n")
    f.write("metrics: \n")
    for metric in model_module.metrics:
        f.write(str(metric)+"\n")
    f.write("Seconds occurred to compile model: "+str(model_module.end-model_module.start)+"\n")
    f.write("using Tensorboard to evaluate training at log dir: "+model_module.log_dir+"\n")
    f.write("\n--FITTING MODEL--\n")
    f.write("callbacks:\n")
    for callback in model_module.callbacks:
        f.write(str(callback)+"\n")
    f.write("monitor metric: "+str(model_module.monitor)+"\n")
    f.write("Transfer Learning history\n")
    for key in history_TL.history.keys():
        f.write(key+" : " + str(model_module.history_TL.history[str(key)])+"\n")
    f.write("Fine Tuning history\n")
    for key in history_FT.history.keys():
        f.write(key+" : " + str(model_module.history_FT.history[str(key)])+"\n")
    f.write("\n--PREDICTIONS MODEL--\n")
    f.write("prediction images are saved at: "+pred_dir+"\n")
    f.write("\n--END--")
    f.close()
