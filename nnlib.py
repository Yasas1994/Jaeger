# import os
# import tensorflow as tf

def rc_cnn_top(x, filters=16, stride=1, kernel_size=5,shared_weights=False,padding='same'):
    x_rc = x[::,::,::-1]
    if shared_weights:
        f = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size, 
                                   strides=stride,kernel_initializer=tf.keras.initializers.HeUniform(),
                                      padding=padding)#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f(x), f(x_rc)

    else:
        f1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,
                                    padding=padding,
                                    strides=stride,
                                    kernel_initializer=tf.keras.initializers.HeUniform())#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        f2 = tf.keras.layers.Conv1D(filters=filters,
                                    padding=padding,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    kernel_initializer=tf.keras.initializers.HeUniform())#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f1(x), f2(x_rc)
        

def rc_cnn_hidden(x_fwd,x_rc, filters=128,kernel_size=5,padding='same', stride=1, shared_weights=True):
    if shared_weights:
        f = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size, strides=stride, kernel_initializer=tf.keras.initializers.HeUniform(),padding=padding)#,activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f(x_fwd), f(x_rc)
    else:
        f1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size, strides=stride,kernel_initializer=tf.keras.initializers.HeUniform(),padding=padding)#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        f2 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size, strides=stride,kernel_initializer=tf.keras.initializers.HeUniform(),padding=padding)#, activity_regularizer=tf.keras.regularizers.l2(l2=1e-5))
        return f1(x_fwd), f2(x_rc)

def rc_conv_block(x_fwd,x_rc, kernel_size=3, filters=16,shared_weights=False,cardinality=12): #simple resnet for viruses#
    # copy tensor to variable called x_skip
    #rc branch
    # Layer 1
    
    fwd_ = []
    rc_ = []
      
    for i in range(cardinality):
        fwd, rc = rc_cnn_hidden(x_fwd, x_rc,filters=4,kernel_size=1,padding='same', shared_weights=shared_weights)
        fwd, rc = rc_batchnorm(x_fwd,x_rc, shared_weights=shared_weights)
        fwd, rc = rc_relu(x_fwd,x_rc)
        fwd, rc = rc_cnn_hidden(x_fwd, x_rc,filters=4,kernel_size=3,padding='same',shared_weights=shared_weights)
        fwd, rc = rc_batchnorm(x_fwd,x_rc,shared_weights=shared_weights)
        fwd, rc = rc_relu(x_fwd,x_rc)  
        fwd, rc = rc_cnn_hidden(x_fwd, x_rc,filters=256,kernel_size=1,padding='same',shared_weights=shared_weights)
        fwd, rc = rc_batchnorm(x_fwd,x_rc,shared_weights=shared_weights)
        fwd_.append(fwd)
        rc_.append(rc)
        
    # Add Residue
    add = tf.keras.layers.Add()
    fwd = add(fwd_)  
    rc = add(rc_) 
    x_fwd = add([fwd,x_fwd])
    x_rc = add([rc,x_rc])
    x_fwd, x_rc = rc_relu(x_fwd,x_rc)
    return x_fwd, x_rc
    

def rc_batchnorm(x_fwd, x_rc, shared_weights=True):
    if shared_weights:
        input_shape = x_rc.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences."
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        out = tf.keras.layers.Concatenate(axis=1)([x_fwd, x_rc])
        out = tf.keras.layers.BatchNormalization()(out)
        split_shape = out.shape[1] // 2
        new_shape = [split_shape, input_shape[2]]
        fwd_out = tf.keras.layers.Lambda(lambda x: x[:, :split_shape, :], output_shape=new_shape)
        rc_out = tf.keras.layers.Lambda(lambda x: x[:, split_shape:, :], output_shape=new_shape)

        x_fwd = fwd_out(out)
        x_rc = rc_out(out)
        #self._current_bn = self._current_bn + 1
        return x_fwd, x_rc
    else:
        bn1 = tf.keras.layers.BatchNormalization()
        bn2 = tf.keras.layers.BatchNormalization()
        
        return bn1(x_fwd), bn2(x_rc)
        

def rc_maxpool(x_fwd, x_rc, pool_size=2):
    mp = tf.keras.layers.MaxPooling1D(pool_size=pool_size)
    return mp(x_fwd), mp(x_rc)

def rc_globalavgpool(x_fwd, x_rc):
    gap = tf.keras.layers.GlobalMaxPooling1D()
    return gap(x_fwd), gap(x_rc)

def rc_resnet_block(x_fwd,x_rc, kernel_size=3, filters=16,shared_weights=False): #simple resnet for viruses#
    # copy tensor to variable called x_skip
    # Layer 1
    fwd, rc = rc_cnn_hidden(x_fwd, x_rc,filters=filters,kernel_size=kernel_size,padding='same', shared_weights=shared_weights)
    fwd, rc = rc_gelu(fwd,rc)
    fwd, rc = rc_batchnorm(fwd,rc, shared_weights=shared_weights)
    
    #layer 2
    fwd, rc = rc_cnn_hidden(fwd, rc,filters=filters,kernel_size=kernel_size,padding='same',shared_weights=shared_weights)
    fwd, rc = rc_gelu(fwd,rc)
    fwd, rc = rc_batchnorm(fwd,rc,shared_weights=shared_weights)
        
    # Add Residue
    add = tf.keras.layers.Add()
    x_fwd = add([x_fwd, fwd])  
    x_rc = add([x_rc, rc])  
    x_fwd, x_rc = rc_gelu(x_fwd,x_rc)
    return x_fwd, x_rc

def rc_dense(x_fwd, x_rc, units=32,shared_weights=True,concat=True,dropout=0.1):
    if shared_weights:
        if concat:
            x=tf.keras.layers.Concatenate()([x_fwd,x_rc])
            if dropout is not None:
                x = tf.keras.layers.Dropout(dropout)(x)
            dense = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.HeUniform())
            return dense(x)
        else:
            dense = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.HeUniform())
            return dense(x_fwd), dense(x_rc)
        
    else:
        dense1 = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.HeUniform())
        dense2 = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.HeUniform())
        return dense1(x_fwd), dense2(x_rc)
        


def rc_relu(x_fwd, x_rc):
    relu = tf.keras.layers.ReLU()
    return relu(x_fwd), relu(x_rc)

def rc_gelu(x_fwd, x_rc):
    gelu = tf.nn.gelu
    return gelu(x_fwd), gelu(x_rc)

def rc_flatten(x_fwd,x_rc):
    
    return tf.keras.layers.Flatten()(x_fwd), tf.keras.layers.Flatten()(x_rc)


###############################models#########################################


#vanila convolution layers can be substituted with depthwiise separable convloutional layers, this reduces the number of computations from 
# filters*kernel_siz*inputdepth to kernel_size*inputdepth + filters*1*inputdepth

def Resnet_12(input_shape, output_shape, return_logits=True, output_bias=None, concat_dense=False, shared_weights=True,global_pooling=True, use_lstm=False,use_dense=True):
    'Resnet12 based model'
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    
    if output_shape == 1:
        output_dense= tf.keras.layers.Dense(1, use_bias=False,kernel_initializer=tf.keras.initializers.GlorotUniform())
        output_activation = tf.keras.activations.sigmoid
     
    else:
        output_dense = tf.keras.layers.Dense(output_shape)
        output_activation = tf.keras.layers.Softmax(axis=-1)   
    
    inputs = tf.keras.Input(shape=input_shape)
    #x = tf.keras.layers.Dropout(0.01, name="input_dropout")(inputs)
    x_fwd, x_rc = rc_cnn_top(inputs, filters=128, stride=1, kernel_size=9, shared_weights=shared_weights)
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)
    x_fwd, x_rc = rc_cnn_hidden(x_fwd,x_rc,filters=128, stride=1, kernel_size=5, shared_weights=shared_weights)
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)
    #x_fwd, x_rc =rc_conv_block(x_fwd,x_rc, kernel_size=3, filters=16,shared_weights=shared_weights,cardinality=12)
    #res = []
    for _ in range(4):
        #res.append(x)
        x_fwd, x_rc = rc_resnet_block(x_fwd,x_rc, kernel_size=5, filters=128, shared_weights=shared_weights)
        #if _ < 3:
        #    x_fwd, x_rc = rc_maxpool(x_fwd,x_rc, pool_size=3)
        #if _ > 0:
        #    x = tf.keras.layers.Add()([res[0], x])
    
    x_concat = tf.keras.layers.Add()([x_rc,x_fwd])
    
    if use_lstm == False:
        
        if global_pooling==True and use_dense == True:
            x_concat = tf.keras.layers.GlobalMaxPool1D()(x_concat)


        elif global_pooling == False:
            x_concat = tf.keras.layers.Flatten()(x_concat) 
            #x_concat = tf.keras.layers.Dense(units=16, kernel_initializer=tf.keras.initializers.HeUniform())(x_concat)
    else:
        
        x_concat = tf.keras.layers.LSTM(units=16)(x_concat)
 

    if use_dense == True:
        
        x_concat = mlp(x_concat, hidden_units=[128,128], dropout_rate=0.5)
        #x_concat= tf.nn.gelu(x_concat)
        x_concat = output_dense(x_concat) #fwd output layer with shared weights
            
    else:
        output_cnn=tf.keras.layers.Conv1D(kernel_size=1,filters=1)
        x_concat = output_cnn(x_concat) #fwd output layer with shared weights
        #rc output with shared weights

    if return_logits:
        return inputs, x_concat
    else:
        output = output_activation(x_concat)
        return inputs, output
    
###################################convtower###################################################

def ConvolutionalTower(inputs,shared_weights=True):
    'Covolutional tower to increase the receptive filed size based'

    x_fwd, x_rc = rc_cnn_top(inputs, filters=128, stride=1, kernel_size=9, shared_weights=shared_weights, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)
    x_fwd, x_rc = rc_cnn_hidden(x_fwd,x_rc,filters=128, stride=1, kernel_size=5, shared_weights=shared_weights, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)
    x_fwd, x_rc = rc_cnn_hidden(x_fwd,x_rc,filters=128, stride=1, kernel_size=5, shared_weights=shared_weights, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)

    for _ in range(4):
        x_fwd, x_rc = (lambda fwd,rc : rc_resnet_block(x_fwd=fwd,x_rc=rc, kernel_size=5, filters=128, shared_weights=shared_weights))(x_fwd,x_rc)
    
    return tf.keras.layers.Add()([x_fwd,x_rc])

###################################convtower2###################################################

def ConvolutionalTower2(inputs,shared_weights=True):
    'Covolutional tower to increase the receptive filed size based'

    x_fwd, x_rc = rc_cnn_top(inputs, filters=128, stride=1, kernel_size=9, shared_weights=shared_weights, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)
    x_fwd, x_rc = rc_cnn_hidden(x_fwd,x_rc,filters=128, stride=1, kernel_size=5, shared_weights=shared_weights, padding='same')
    x_fwd, x_rc = rc_gelu(x_fwd, x_rc)
    x_fwd, x_rc = rc_batchnorm(x_fwd,x_rc)
    x_fwd, x_rc = rc_maxpool(x_fwd, x_rc,pool_size=2)

    for _ in range(3):
        x_fwd, x_rc = (lambda fwd,rc : rc_resnet_block(x_fwd=fwd,x_rc=rc, kernel_size=5, filters=128, shared_weights=shared_weights))(x_fwd,x_rc)
    
    return tf.keras.layers.Add()([x_fwd,x_rc])

#####################################LSTM##############################################

def LSTM_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x=ConvolutionalTower(inputs)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    out = tf.keras.layers.Dense(4)(x)
    return inputs,out

###################################vitra###############################################
class Patches(tf.keras.layers.Layer):
    
    def __init__(self, num_patches, patch_size, name = "split"):
        super(Patches, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.layer_name = name

    def call(self, data):
        batch_size = tf.shape(data)[0]
        splitted_seq = tf.split(data, num_or_size_splits=self.num_patches, axis=1 ,num=self.patch_size, name=self.layer_name)
        patches = tf.stack( splitted_seq , axis=1, name='stack')
        patches = tf.reshape(patches, [batch_size,self.num_patches ,self.patch_size*4])
       
        ##patch_dims = patches.shape[-1]
        ##patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        
        return patches

    

class PatchEncoder(tf.keras.layers.Layer): #Parch encoding + Position encoding 
    def __init__(self, num_patches, projection_dim=None,embed_input=False,use_sine=True): #num_patches == sequence length when input comes from a conv block
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        
        if embed_input is True:
            self.projection = tf.keras.layers.Dense(units=projection_dim)
        else:
            self.projection = None
            
        if use_sine == False:
            self.position_embedding_layer = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )
        else:
            position_embedding_matrix = self.get_position_encoding(num_patches, projection_dim)                                          
        
            self.position_embedding_layer = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim,
                weights=[position_embedding_matrix],
                trainable=False
            )
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        if self.projection is not None:
            input_projection=self.projection(patches)
        else:
            input_projection = patches
            
        encoded = input_projection + self.position_embedding_layer(positions)
        
        
        return encoded
    
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def Vitra(input_shape=(2048,4,),num_patches=512,transformer_layers = 4,num_heads=4,  att_dropout=0.1,
                          projection_dim=128, att_hidden_units=[128,128],mlp_hidden_units=[128,128],
                          mlp_dropout=0.1, use_global=True, global_type='max'):
    inputs = tf.keras.Input(shape=input_shape)
    # Create patches.
    patches=ConvolutionalTower2(inputs)
    #patches = Patches(num_patches=num_patches,patch_size=patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=att_dropout
        )(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=att_hidden_units, dropout_rate=mlp_dropout)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    if use_global == True:
        if global_type == 'average':
            representation=tf.keras.layers.GlobalAveragePooling1D()(representation)
        elif global_type == 'max':
            representation = tf.keras.layers.GlobalMaxPooling1D()(representation)
    else:
        representation = tf.keras.layers.Flatten()(representation)
        
    representation = tf.keras.layers.Dropout(0.1)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_hidden_units, dropout_rate=0.5)
    # Classify outputs.
    logits = tf.keras.layers.Dense(4)(features)
    # Create the Keras model.
    return inputs, logits
