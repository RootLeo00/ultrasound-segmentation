import datetime

from params import IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, MODEL_NAME, base_dir,pred_dir, model_dir, train_dir, test_dir, description, weights_path
from utils.load_dataset import dataset_params
from predict import pred_params
from train import train_params

def print_only_predict_details():
    """
    Prints details of the model into a file of only predict program.
    """

    f = open(pred_dir+"/prediction" +
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".txt", "a")
    f.write("--START--\n")
    f.write("Using dataset at: "+base_dir+"\n")
    f.write("IMAGE SIZE: "+str(IMAGE_SIZE)+"\n")
    f.write("NUMBER OF CLASSES: "+str(NUM_CLASSES)+"\n")
    f.write("BATCH SIZE: "+str(BATCH_SIZE)+"\n")
    f.write("NUMBER OF EPOCHS: "+str(EPOCHS)+"\n")

    f.write("****COMMENT****\n")
    f.write(description)
    f.write("\n*********\n")

    f.write("Using "+str(dataset_params['num_train_images']) +
            " train images loaded from: "+train_dir+"\n")
    f.write("Using "+str(dataset_params['num_test_images']) +
            " test images are loaded from: "+test_dir+"\n")
    f.write("Shape of test images: "+ str(pred_params['shape_test_imgs'])+"\n")
    f.write("Shape of test masks: "+ str(pred_params['shape_test_imgs'])+"\n")
    f.write("The model used for this prediction is: " + MODEL_NAME+"\n")
    f.write("Model loaded with weights from: " +weights_path +"\n")
    
    f.write("\n--PREDICTIONS MODEL--\n")
    f.write("prediction images and graphs are saved at: "+pred_dir+"\n")
    f.write("\n--END--")
    f.close()



def print_train_details():
    """
    Prints details of the model into a file of only predict program.
    """

    f = open(pred_dir+"/prediction" +
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".txt", "a")
    f.write("--START--\n")
    f.write("Using dataset at: "+base_dir+"\n")
    f.write("IMAGE SIZE: "+str(IMAGE_SIZE)+"\n")
    f.write("NUMBER OF CLASSES: "+str(NUM_CLASSES)+"\n")
    f.write("BATCH SIZE: "+str(BATCH_SIZE)+"\n")
    f.write("NUMBER OF EPOCHS: "+str(EPOCHS)+"\n")

    f.write("****COMMENT****\n")
    f.write(description)
    f.write("\n*********\n")

    f.write("Using "+str(dataset_params['num_test_images']) +
            " test images are loaded from: "+test_dir+"\n")
    f.write("Shape of test images: "+     str(pred_params['shape_test_imgs'])+"\n")
    f.write("Shape of test masks: "+     str(pred_params['shape_test_imgs'])+"\n")
    f.write("The model used for this training is: " + MODEL_NAME+"\n")
    f.write("Model summary:\n")
    train_params['model'].summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n\n--COMPILE, TRAIN AND EVALUATE MODEL--\n")
    f.write("optimizer: "+str(train_params['optimizer'])+"\n")
    f.write("loss: " + str(train_params['loss']) + "\n")
    f.write("with class_weights: "+str(train_params['class_weights'])+"\n")
    f.write("metrics: \n")
    for metric in train_params['metrics']:
        f.write(str(metric)+"\n")
    f.write("using Tensorboard to evaluate training at log dir: "+train_params['tensorboard_log_dir']+"\n")
    f.write("\n\n--FITTING MODEL--\n")
    f.write("callbacks:\n")
    for callback in train_params['callbacks']:
        f.write(str(callback)+"\n")
    f.write("patience earlystopping: "+str(train_params['patience'])+"\n")
    f.write("monitor metric: "+str(train_params['monitor'])+"\n")
    f.write("time to train (fit) transfer learning: "+str(train_params['timeTL'])+"\n")
    f.write("time to train (fit) fine tuning: "+str(train_params['timeFT'])+"\n")
    f.write("\n\nTransfer Learning history\n")
    for key in train_params['history_TL'].keys():
        f.write(key+" : " + str(train_params['history_TL'][str(key)])+"\n")
    f.write("\n\nFine Tuning history\n")
    for key in train_params['history_FT'].keys():
        f.write(key+" : " + str(train_params['history_FT'][str(key)])+"\n")
    f.write("\n\n--PREDICTIONS MODEL--\n")
    f.write("prediction images and graphs are saved at: "+pred_dir+"\n")
    f.write("\n--END--")
    f.close()
