import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.backend as K
from tensorflow.data import Dataset, TextLineDataset, TFRecordDataset
import numpy as np
import nnlib
from confreader import Configuration
from utils import DataPaths



if __name__ == "__main__":
    
    seed=np.random.randint(100000)
    tf.random.set_seed(seed)
    
    #load configuration file
    conf = Configuration.load_json('config.json') #loads the configuration file
    
    #load data
    data_dirs = DataPaths().update_data_paths(dir_positive='../Data/Bacteria/phaster/1000/',
                                            dir_negative='../Data/Bacteriophages/inphared+imgvrsin/1000_new/',
                                            suf_neg=None,suf_pos=None)
 


       
    #build model
    inputs, outputs = LSTM_model(input_shape=(2048,4))
    model = CustomModel(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.optimizer.initial_lr,clipnorm=1)

    model.compile(optimizer=optimizer)
    
    try:
        model.save_weights(checkpoint_path.format(epoch=0))
    except:
        print('checkpoint path is not set, failed to save intial weights')
        
    #load training data
    dataset = TextLineDataset(train_paths, num_parallel_reads=10, buffer_size=2000000)
    data=dataset.map(process_string, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(32, drop_remainder=True)
    #load validation data
    dataset_val = TextLineDataset(val_paths, num_parallel_reads=4, buffer_size=2000000)
    data_val=dataset_val.map(process_string, num_parallel_calls=tf.data.AUTOTUNE).batch(512)
    #loading test data
    dataset_test = TextLineDataset(test_paths[-1], num_parallel_reads=4 , buffer_size=20000)
    data_test=dataset_test.map(process_string, num_parallel_calls=4).batch(512, drop_remainder=True)
        
    #train model
    model.fit(
                data,
                epochs=10,
                initial_epoch = 0,
                validation_data=data_val.take(100),
                callbacks=[EarlyStopping, ModelCheckPoint, CSVLogger, LRScheduler,ReduceLr ])