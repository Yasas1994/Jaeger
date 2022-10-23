import os
import tensorflow as tf

def rc_cnn(x, name='', filters=16, stride=1, kernel_size=5,dilation_rate=1,padding='same'):

    f = tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=kernel_size, 
                               dilation_rate=dilation_rate,
                               strides=stride,
                               kernel_initializer=tf.keras.initializers.HeUniform(),
                               padding=padding,
                               name=name)#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
    outputs = []
    for l in x:
        outputs.append(f(l))
        
    return outputs


    
def rc_batchnorm(x, name):


    f = tf.keras.layers.BatchNormalization(name=f'bn_{name}')
    
    outputs = []
    for l in x:
        outputs.append(f(l))
        
    return outputs

        

def rc_maxpool(x, pool_size=2):
    f = tf.keras.layers.MaxPooling1D(pool_size=pool_size)
    outputs = []
    for l in x:
        outputs.append(f(l))
        
    return outputs


def rc_gelu(x):
    f = tf.nn.gelu
    outputs = []
    for l in x:
        outputs.append(f(l))
        
    return outputs

def rc_resnet_block(x, name, kernel_size=[3,3],dilation_rate=[1,1], filters=[16,16],shared_weights=False, add_residual=True): #simple resnet for viruses#
    '''x: input tensor
       name:name for the block
       kernel_size: a list specifying the kernel size of each conv layer
       dilation_rate: a list specifying the dilation rate of each conv layer
       filter:  a list specifying the number of filters of each conv layer
       shared_weights: whether to use reverse conplement parameter sharing.(True)
       add_residual: whether to add residual connections.(True)
    '''

    xx = rc_cnn(x,
                name=f'{name}{1}',
                filters=filters[0],
                kernel_size=kernel_size[0],
                padding='same',
                dilation_rate=dilation_rate[0])
    xx= rc_gelu(xx)
    xx = rc_batchnorm(xx,name=f'{name}{1}')
    # Create layers
    for n, (k, d, f) in enumerate(zip(kernel_size[1:], dilation_rate[1:], filters[1:])):
        xx = rc_cnn(xx,
                                name=f'{name}{n+2}',
                                filters=f,
                                kernel_size=k,
                                padding='same',
                                dilation_rate=d)
        xx = rc_gelu(xx)
        xx = rc_batchnorm(xx,name=f'{name}{n+2}')
    

    #scale up the skip connection output if the filter sizes are different 
    
    if (filters[-1] != filters[0]  or x[-1].shape[-1] != filters[-1]) and add_residual:
        x = rc_cnn(x,
                    name=f'{name}_skip',
                    filters=f,
                    kernel_size=1,
                    padding='same',
                    dilation_rate=1)
        x = rc_gelu(x)
        x = rc_batchnorm(x,name=f'{name}_skip')
        
    # Add Residue
    outputs = []
    add = tf.keras.layers.Add()
    if add_residual:
        for l in zip(x,xx):
            outputs.append( add(l))  
        
        
    return rc_gelu(outputs)


def ConvolutionalTower(inputs):
    'Covolutional tower to increase the receptive filed size based on dilated convolutions'

    x = rc_cnn(inputs, filters=128, stride=1, kernel_size=9, dilation_rate=1, padding='same')
    x = rc_gelu(x)
    x = rc_batchnorm(x,name='block1_1')
    x = rc_maxpool(x,pool_size=2)
    x = rc_cnn(x,name='block1_1',filters=128, stride=1, kernel_size=5, dilation_rate = 2,padding='same')
    x = rc_gelu(x)
    x = rc_batchnorm(x, name='block1_2')
    x = rc_maxpool(x,pool_size=2)

    for i,n in enumerate(range(5)):
        x = (lambda x,n,i : rc_resnet_block(x,
                                         name=f'block2_{n}',
                                         kernel_size=[5,5],
                                         dilation_rate=[3+i,3+i], 
                                         filters=[128,128]))(x,n,i)
    
    return tf.keras.layers.Add()(x)


def LSTM_model(input_shape=None): #archeae model 1
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22, 4, name="codon")
    embeddings = []
    for l in [f1input,f2input,f3input,r1input,r2input,r3input]:
        embeddings.append(embedding_layer(l))
        
    x=ConvolutionalTower(embeddings)
    #x=tf.keras.layers.GlobalMaxPool1D()(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, name='lstm'),name='bidirlstm')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu, name='augdense-1')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu,name='augdense-2')(x)
#     x = tf.keras.layers.Dropout(0.1)(x)
#     x = tf.keras.layers.Dense(128, activation=tf.nn.gelu,name='augdense-3')(x)
    out = tf.keras.layers.Dense(4,name='outdense')(x)
    return [f1input,f2input,f3input,r1input,r2input,r3input],out
