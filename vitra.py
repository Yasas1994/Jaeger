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
