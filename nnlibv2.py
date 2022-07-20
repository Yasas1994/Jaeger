import os
import tensorflow as tf

def rc_cnn_top(x, filters=16, stride=1, kernel_size=5,shared_weights=False,dilation_rate=1,padding='same',drop=None):
    if drop:
        x=tf.keras.layers.Dropout(drop,name='rcconv1_dropout')(x)
    x_rc = x[::,::,::-1]
    if shared_weights:
        f = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size, 
                                   dilation_rate=dilation_rate,
                                   strides=stride,
                                   kernel_initializer=tf.keras.initializers.HeUniform(),
                                   padding=padding,
                                   name='rcconv1_top')#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f(x), f(x_rc)

    else:
        f1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,
                                    padding=padding,
                                    dilation_rate=dilation_rate,
                                    strides=stride,
                                    kernel_initializer=tf.keras.initializers.HeUniform(),
                                    name='rcconv1_top')#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        f2 = tf.keras.layers.Conv1D(filters=filters,
                                    padding=padding,
                                    kernel_size=kernel_size,
                                    dilation_rate=dilation_rate,
                                    strides=stride,
                                    kernel_initializer=tf.keras.initializers.HeUniform(),
                                    name='rcconv2_top')#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f1(x), f2(x_rc)
        

def rc_cnn_hidden(x_fwd,x_rc,name, filters=128,kernel_size=5,padding='same', dilation_rate=1,stride=1, shared_weights=True):
    if shared_weights:
        f = tf.keras.layers.Conv1D(filters=filters,
                                   dilation_rate=dilation_rate,
                                   kernel_size=kernel_size, 
                                   strides=stride, 
                                   kernel_initializer=tf.keras.initializers.HeUniform(),
                                   padding=padding,
                                   name=f'rcconv_{name}')#,activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f(x_fwd), f(x_rc)
    else:
        f1 = tf.keras.layers.Conv1D(filters=filters,
                                    dilation_rate=dilation_rate,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    kernel_initializer=tf.keras.initializers.HeUniform(),
                                    padding=padding,
                                    name=f'rcconv_fwd_{name}'
                                   )#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        f2 = tf.keras.layers.Conv1D(filters=filters,
                                    dilation_rate=dilation_rate,
                                    kernel_size=kernel_size, 
                                    strides=stride,
                                    kernel_initializer=tf.keras.initializers.HeUniform(),
                                    padding=padding,
                                    name=f'rcconv_rc_{name}')#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f1(x_fwd), f2(x_rc)

    

def rc_batchnorm(x_fwd, x_rc, name, shared_weights=True):
    if shared_weights:
        input_shape = x_rc.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output"
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        out = tf.keras.layers.Concatenate(axis=1)([x_fwd, x_rc])
        out = tf.keras.layers.BatchNormalization(name=f'bn_{name}')(out)
        split_shape = out.shape[1] // 2
        new_shape = [split_shape, input_shape[2]]
        fwd_out = tf.keras.layers.Lambda(lambda x: x[:, :split_shape, :], output_shape=new_shape)
        rc_out = tf.keras.layers.Lambda(lambda x: x[:, split_shape:, :], output_shape=new_shape)

        x_fwd = fwd_out(out)
        x_rc = rc_out(out)
        #self._current_bn = self._current_bn + 1
        return x_fwd, x_rc
    else:
        bn1 = tf.keras.layers.BatchNormalization(name=f'bn_fwd{name}')
        bn2 = tf.keras.layers.BatchNormalization(name=f'bn_rc{name}')
        
        return bn1(x_fwd), bn2(x_rc)
        

def rc_maxpool(x_fwd, x_rc, pool_size=2):
    mp = tf.keras.layers.MaxPooling1D(pool_size=pool_size)
    return mp(x_fwd), mp(x_rc)

def rc_globalavgpool(x_fwd, x_rc):
    gap = tf.keras.layers.GlobalMaxPooling1D()
    return gap(x_fwd), gap(x_rc)


def rc_dense(x_fwd, x_rc,name, units=32,shared_weights=True,concat=True,dropout=0.1):
    if shared_weights:
        if concat:
            x=tf.keras.layers.Concatenate()([x_fwd,x_rc])
            if dropout is not None:
                x = tf.keras.layers.Dropout(dropout)(x)
            dense = tf.keras.layers.Dense(units,
                                          name=f'dense_{name}',
                                          kernel_initializer=tf.keras.initializers.HeUniform())
            return dense(x)
        else:
            dense = tf.keras.layers.Dense(units,
                                          name=f'dense_{name}',
                                          kernel_initializer=tf.keras.initializers.HeUniform())
            return dense(x_fwd), dense(x_rc)
        
    else:
        dense1 = tf.keras.layers.Dense(units,
                                       name=f'dense_{name}1',
                                       kernel_initializer=tf.keras.initializers.HeUniform())
        dense2 = tf.keras.layers.Dense(units,
                                       name=f'dense_{name}2',
                                       kernel_initializer=tf.keras.initializers.HeUniform())
        return dense1(x_fwd), dense2(x_rc)
        


def rc_relu(x_fwd, x_rc):
    relu = tf.keras.layers.ReLU()
    return relu(x_fwd), relu(x_rc)

def rc_gelu(x_fwd, x_rc):
    gelu = tf.nn.gelu
    return gelu(x_fwd), gelu(x_rc)

def rc_flatten(x_fwd,x_rc):
    
    return tf.keras.layers.Flatten()(x_fwd), tf.keras.layers.Flatten()(x_rc)




def rc_resnet_block(x_fwd,x_rc, name, kernel_size=[3,3],dilation_rate=[1,1], filters=[16,16],shared_weights=False, add_residual=True): #simple resnet for viruses#
    '''x_fwd:forward strand feature tensor
       x_rc:reverse strand feature tensor
       name:name for the block
       kernel_size: a list specifying the kernel size of each conv layer
       dilation_rate: a list specifying the dilation rate of each conv layer
       filter:  a list specifying the number of filters of each conv layer
       shared_weights: whether to use reverse conplement parameter sharing.(True)
       add_residual: whether to add residual connections.(True)
    '''

    fwd, rc = rc_cnn_hidden(x_fwd, x_rc,
                            name=f'{name}{1}',
                            filters=filters[0],
                            kernel_size=kernel_size[0],
                            padding='same',
                            dilation_rate=dilation_rate[0],
                            shared_weights=shared_weights)
    fwd, rc = rc_gelu(fwd,rc)
    fwd, rc = rc_batchnorm(fwd,rc,name=f'{name}{1}',shared_weights=shared_weights)
    # Create layers
    for n, (k, d, f) in enumerate(zip(kernel_size[1:], dilation_rate[1:], filters[1:])):
        fwd, rc = rc_cnn_hidden(fwd, rc,
                                name=f'{name}{n+2}',
                                filters=f,
                                kernel_size=k,
                                padding='same',
                                dilation_rate=d,
                                shared_weights=shared_weights)
        fwd, rc = rc_gelu(fwd,rc)
        fwd, rc = rc_batchnorm(fwd,rc,name=f'{name}{n+2}',shared_weights=shared_weights)
    

    #scale up the skip connection output if the filter sizes are different 
    if filters[-1] > filters[0] and add_residual:
        x_fwd, x_rc = rc_cnn_hidden(x_fwd, x_rc,
                                    name=f'{name}_skip',
                                    filters=f,
                                    kernel_size=1,
                                    padding='same',
                                    dilation_rate=1,
                                    shared_weights=shared_weights)
        x_fwd, x_rc = rc_gelu(x_fwd,x_rc)
        x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc, name=f'{name}_skip',
                                   shared_weights=shared_weights)
        
    # Add Residue
    if add_residual:
        add = tf.keras.layers.Add()
        fwd = add([x_fwd, fwd])  
        rc = add([x_rc, rc])  
        
    fwd, rc = rc_gelu(fwd,rc)
    return fwd, rc


##########################################################################


def ConvolutionalTower(inputs,shared_weights=True):
    'Covolutional tower to increase the receptive filed size based'

    x_fwd, x_rc = rc_cnn_top(inputs, filters=128, stride=1, kernel_size=9, shared_weights=shared_weights, dilation_rate=2, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc,name='block1_1')
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)
    x_fwd, x_rc = rc_cnn_hidden(x_fwd,x_rc,name='block1_1',filters=128, stride=1, kernel_size=5, dilation_rate = 3,shared_weights=shared_weights, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc, name='block1_2')
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)

    for _ in range(5):
        x_fwd, x_rc = (lambda fwd,rc,n : rc_resnet_block(x_fwd=fwd,x_rc=rc,
                                                         name=f'block2_{n}',
                                                         kernel_size=[5,5],
                                                         dilation_rate=[3,3], 
                                                         filters=[128,128], 
                                                         shared_weights=shared_weights))(x_fwd,x_rc,_)
    
    return tf.keras.layers.Add()([x_fwd,x_rc])


###########################################################################

def LSTM_model(input_shape): #archeae model 1
    inputs = tf.keras.Input(shape=input_shape)
    x=ConvolutionalTower(inputs)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    out = tf.keras.layers.Dense(5)(x)
    return inputs,out


