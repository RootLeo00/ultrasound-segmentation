
# Function for fine-tuning the model (unfreeze all layers) 
def finetune_unfreezeall(input_shape, base_model):
    input_shape = input_shape
    base_model = base_model
    
    # unfreeze the contracting path and freeze the expanding path
    for layer in base_model.layers:
        if layer.name in ['block1_conv1', 'block1_conv2', 'block1_pool',
                          'block2_conv1', 'block2_conv2', 'block2_pool', 
                          'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
                          'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                          'block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True
            
    return base_model    

# Function for fine-tuning the model (unfreeze deepest layers) 
def finetune_unfreezedeepest(input_shape, base_model):
    input_shape = input_shape
    base_model = base_model
    
    # unfreeze the contracting path and freeze the expanding path
    for layer in base_model.layers:
        if layer.name in ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                          'block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True
            
    return base_model    

