import os
import io
import sys
import argparse
import logging
import types
import pkg_resources

from Bio import SeqIO
from Bio.Seq import Seq
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.backend as K

#from tensorflow.data import Dataset, TextLineDataset, TFRecordDataset
import numpy as np
from jaegeraa.nnlib.layers import WRes_model
from jaegeraa.nnlib.cmodel import JaegerModel
from jaegeraa.utils import get_compressed_file_handle
from jaegeraa.preprocessing import fasta_gen_lib, codon_mapper, process_string, c_mapper
from jaegeraa.postprocessing import extract_pred_entry, per_class_preds, average_per_class_score, get_class, pred2string


#write a new fasta_gen that accepts strings, file handles and SeqIO objects
class Predictor:
    def __init__(self):
        
        self.stratergy = tf.distribute.MirroredStrategy()
        self.weights_root = pkg_resources.resource_filename('jaegeraa', 'data/')
        self.weights_path = os.path.join(self.weights_root,'WRes_1024.h5')
        with self.stratergy.scope():
            self.inputs, self.outputs = WRes_model(input_shape=(None,))
            self.model = JaegerModel(inputs=self.inputs, outputs=self.outputs)
            self.model.load_weights(filepath=self.weights_path)#.expect_partial() when loading weights from a chpt file
    
    def predict(self,input,stride=2048,fragsize=2048,batch=100):
        results = {'contig_id':[],'length':[],'#num_prok_windows':[],'#num_vir_windows':[],'#num_fun_windows':[],
                    '#num_arch_windows':[],'prediction':[],'bac_score':[],'vir_score':[], 'fun_score':[],'arch_score':[], 
                    'window_summary':[]}

        with tf.device('/cpu:0'):
            input_dataset = tf.data.Dataset.from_generator(fasta_gen_lib(input,fragsize=fragsize,stride=stride),
                                                        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)))
            idataset=input_dataset.map(process_string(crop_size=fragsize,),
                                num_parallel_calls=tf.data.AUTOTUNE,).batch(batch_size=batch, num_parallel_calls=tf.data.AUTOTUNE)
            for c,(a,s,d,f,l) in enumerate(extract_pred_entry(self.model,idataset)):
                #code for further processing goes here
                c1 = per_class_preds(s)
                c4 = average_per_class_score(a)
                c2 = get_class(c4, False)
                c5 = pred2string(s)

                results['contig_id'].append(d[-1].decode())
                results['length'].append(l[-1])
                results['#num_prok_windows'].append(c1[0])
                results['#num_vir_windows'].append(c1[1])
                results['#num_fun_windows'].append(c1[2])
                results['#num_arch_windows'].append(c1[3])
                results['prediction'].append(c2[0])
                results['bac_score'].append(c4[0])
                results['vir_score'].append(c4[1])
                results['fun_score'].append(c4[2])
                results['arch_score'].append(c4[3])
                results['window_summary'].append(c5)
        return results
