'''/**
 * Jaeger project - Identification of viruses in metagenomes 
                    using deep learning
 * File: inference.py; inference programs for the karas model
 * User: Y. Wijesekara
 * Date: 1/6/22
 * Time: 12:54 PM
 * Email: yasas.wijesekara@uni-greifswald.de
 */'''


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from Bio import SeqIO
import tensorflow as tf
#tf.get_logger().setLevel('INFO')
#tf.config.threading.set_inter_op_parallelism_threads(num_threads=20)
#tf.config.threading.set_intra_op_parallelism_threads(num_threads=20)
#tf.config.set_soft_device_placement(True)
import tensorflow.keras as keras
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.backend as K
from tensorflow.data import Dataset, TextLineDataset, TFRecordDataset
import numpy as np
from nnlibv2 import LSTM_model
from cmodel import CustomModel
from utils import DataPaths
import argparse
from tqdm import tqdm
from typing import Iterable, Any, Tuple
from enum import Enum, auto
import bz2
import gzip
import logging
import lzma

###############################utils######################################
def signal_fl(it:Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    '''get a signal at the begining and the end of a iterator'''
    iterable = iter(it)
    yield True, next(iterable)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var
def signal_l(it:Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    '''get a signal at the end of the iterator'''
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var
    
class Compression(Enum):
    gzip = auto()
    bzip2 = auto()
    xz = auto()
    noncompressed = auto()
    
def is_compressed(filepath):
    """ Checks if a file is compressed (gzip, bzip2 or xz) """
    with open(filepath, "rb") as fin:
        signature = fin.peek(8)[:8]
        if tuple(signature[:2]) == (0x1F, 0x8B):
            return Compression.gzip
        elif tuple(signature[:3]) == (0x42, 0x5A, 0x68):
            return Compression.bzip2
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            return Compression.xz
        else:
            return Compression.noncompressed
def get_compressed_file_handle(path):
    filepath_compression = is_compressed(path)
    if filepath_compression == Compression.gzip:
        f = gzip.open(path, "rt")
    elif filepath_compression == Compression.bzip2:
        f = bz2.open(path, "rt")
    elif filepath_compression == Compression.xz:
        f = lzma.open(path, "rt")
    else:
        f = open(path, "r")
    return f
###################input pre-processing#################################
def mapper():
    '''creates a hashtable which will later be used by tensorflow to 
    convert string vectors to one-hot vectors
    returns : Hashtable
    '''
    keys_tensor = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
    vals_tensor = tf.constant([0,3,1,2,0,3,1,2])
    init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    table = tf.lookup.StaticHashTable(init, default_value=4)
    return table


#new functions
def fasta_gen(filehandle,fragsize=None,stride=None,num=None): #fasta sequence generator
    #filename here is a reference to a file handle
    #should also be able to handle small sequences
    #simply remove short sequnces, write to a separate file 
    def c():
        #accepts a reference to a file handle
        
            for record in tqdm(SeqIO.FastaIO.SimpleFastaParser(filehandle), total=num, bar_format='{l_bar}{bar:80}{r_bar}{bar:-10b}'):
                seqlen=len(record[1])
                if seqlen >= fragsize:
                    if fragsize is None: #if no fragsize, return the entire sequence 
                        yield str(record[1])+","+str(record[0]) #sequence and sequence headder 
                    elif fragsize is not None:
                        for i,(l,index) in enumerate(signal_l(range(0,seqlen-(fragsize-1),fragsize if stride is None else stride))):
                            yield str(record[1])[index:index+fragsize]+","+str(record[0].replace(',',''))+","+str(index)+","+str(l)+","+str(i)+","+str(seqlen)
    return c #return fasta sequence generator
#@tf.function
def process_input_string(table=None, onehot=True):
    def x(string,table=table, onehot=onehot):
        x = tf.strings.split(string, sep=',')
        s = tf.strings.bytes_split(x[0]) 
        f=table.lookup(s) #convert string to int vector
        #x2 = index, x3=if last
        f2 = x[1] #label or sequence headder. use this to traceback

        if onehot:
            return tf.one_hot(f, depth=4, dtype=tf.float16, on_value=0.99, off_value=0.033),f2,x[2],x[3],x[4],x[5]#return onh 
        else:
            return f #retruns int vector
    return x

def fragen(fasta_file,fragsize=2048, stride=1024):
    '''returns a nucleotide fragment generator '''
    def c():
        for index in range(0,len(record.seq)-fragsize,fragsize if stride is None else stride):
            yield str(record.seq)[index:index+fragsize]+","+str(record.id)
    return c
######################################calling the model########################################
def get_predictions(idataset):  #get predictions per batch  

    for batch in idataset:
        probs= tf.keras.activations.softmax(model(batch[0])) #convert logits to probabilities 
        #probsperpos.append(probs) #probabilities per position
        y_pred= tf.argmax(probs,-1) #get predicted class for each instance in the batch   
        yield probs, y_pred, batch[1], batch[2], batch[3], batch[4], batch[5]
        
def ExtractPredPEntry(idataset, numclass=5):#takes a generator as input 
    '''
    #prob : position wise softmax prbs
    #y_pred : position wise predicted classes
    #id_: fasta headder
    #pos_:position in the sequence
    #index_:index at pos_[i]
    #is_last:end of sequence signal'''
    
    tmp_prob = np.empty((0,numclass),float) #empty np.array to store predicted probabilities
    tmp_ypred = np.empty((0),int) #empty np.array to store predicted classes
    tmp_id = [] #contig id 
    tmp_pos = [] #genomic cordinate vector -> used for prophage prediction 
    tmp_len = []
    
    for prob,y_pred,id_,pos_,is_last_,index_,clen_ in get_predictions(idataset):
        #
        #prb_tmp.extend([i.numpy() for i in prob])
        #print(prb_tmp)
        tmp_prob = np.append(tmp_prob, prob, axis=0) #concatenate the prob vectors 
        tmp_ypred = np.append(tmp_ypred, y_pred, axis=0) #concatenate the prob vectors 
        tmp_id.extend([i for i in id_.numpy()]) #adds contig ids
        tmp_pos.extend([int(i) for i in pos_.numpy()])
        tmp_len.extend([int(i) for i in clen_.numpy()])
        
        for index,is_last in zip(index_,is_last_): #when is_last signal is received, return the prob vector
    
            if int(is_last) == 1:
                #print(prb_tmp.shape)
                tmp = tmp_prob[:int(index)+1]
                tmp_prob = tmp_prob[int(index)+1:]
                
                tmp1 = tmp_ypred[:int(index)+1]
                tmp_ypred = tmp_ypred[int(index)+1:]
                
                tmp2   = tmp_id[:int(index)+1]
                tmp_id = tmp_id[int(index)+1:]
                
                tmp3   = tmp_pos[:int(index)+1]
                tmp_pos = tmp_pos[int(index)+1:]
                
                tmp4   = tmp_len[:int(index)+1]
                tmp_len = tmp_len[int(index)+1:]
                
                #print(prb_tmp.shape, tmp.shape)
                yield tmp, tmp1, tmp2, tmp3, tmp4
    
def PerClassPreds(y_pred,numclass=5): #y_pred is a vector with scores for the entire entry
    return np.array([np.sum(y_pred==i) for i in range(numclass)])

def AveragePerClassScore(yprob):
    return np.mean(yprob, axis=0)

def GetClass(y_pred, get_all_classes=False): 
    y_pred=y_pred/sum(y_pred)
    c=np.argmax(y_pred)
    if c == 0:
        return "Prokaryota" if get_all_classes else "Non-viral",round(y_pred[c],3)
    elif c == 1:
        return "Virus",round(y_pred[c],3)
    elif c == 2:
        return "Fungi" if get_all_classes else "Non-viral",round(y_pred[c],3)
    elif c==3:
        return "Protista" if get_all_classes else "Non-viral",round(y_pred[c],3)
    elif c==4:
        return "Archaea" if get_all_classes else "Non-viral",round(y_pred[c],3)
    else:
        raise "class can not be > 5"

def get_hs_positions(probsperpos, numclass=5):    
    '''extracts high scoring positions from each class
    return a dict'''
    y_preds=np.argmax(probsperpos,axis=1)
    #get high scoring positions for each class
    scorepos={class_:np.where(np.argmax(probsperpos,axis=1)==class_)[0] for class_ in range(numclass)}
    return scorepos

def get_perclass_probs(probperpos, numclass=5):
    '''creates a per class probabiliy dict'''
    return {i: probperpos[::,i] for i in range(numclass) }

def pred2string(predictions):
    '''
    converts preidctions along the sequence to a string
    predictions: vector of predicted classes for all windows'''
    string = ''
    tmp_N = 0
    tmp_V = 0
    for j,i in  enumerate(predictions):
        if i == 1:
            if j > 0 and tmp_N >0:
                string += f'{tmp_N}n'
                tmp_N = 0
            tmp_V += 1 
            
        else:
            if j > 0 and tmp_V > 0:
                string += f'{tmp_V}V'
                tmp_V= 0
            tmp_N += 1
    if tmp_V > 0:
        string += f'{tmp_V}V'
    elif tmp_N > 0:
        string += f'{tmp_N}n'
    return string 

###########################################statistics & plots###################################

            

##########################################argument-parser#######################################
def dir_path(path):
    '''checks path and creates if absent'''
    if os.path.isdir(path):
        return path
    else:
        os.mkdir(path)
        return path


def file_path(path):
    '''checks if file is present'''
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"ERROR:{path} is not a valid file")

def cmdparser():
    '''cmdline argument parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument("-i","--input",
                        type=file_path,
                        required=True,
                        help="path to input file")
    parser.add_argument("-o","--output", 
                        type=str,
                        required=True,
                        help='path to output file')
    parser.add_argument("-f","--format", 
                        type=str,
                        nargs='?',
                        default="fasta", 
                        help="input file format, default: fasta ('genbank','fasta')")
    parser.add_argument("--stride", 
                        type=int,
                        nargs='?',
                        default=2048, 
                        help="stride of the sliding window. default:2048")
    parser.add_argument("--batch", 
                        type=int,
                        nargs='?',
                        default=2048, 
                        help="parallel batch size, set to a lower value if your gpu runs out of memory. default:2048")
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--cpu", 
                        action="store_true",
                        help="Number of threads to use for inference. default:False")
    group.add_argument("--gpu",
                        action="store_false",
                        help='run on gpu, runs on all gpus on the system by default: default: True')

    parser.add_argument("--getalllabels", 
                        action="store_true",
                        help="get predicted labels for Non-Viral contigs. default:False")
    parser.add_argument("--meanscore",
                        action="store_true", 
                        help="output mean predictive score per contig. deafault:True")
    parser.add_argument("--fragscore", 
                        action="store_true", 
                        help="output percentage of perclass predictions per contig. deafault:True")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="increase output verbosity")
    
    
    return parser.parse_args()

################################################################################################

if __name__ == "__main__":

    def get_gpu_count():
        gpus=0
        for i in tf.config.get_visible_devices():
            if 'GPU' in i.device_type:
                gpus += 1
        return gpus

    def configure_multi_gpu_inference(gpus):
        if gpus > 0:
             return tf.distribute.MirroredStrategy()
        else:
            None
            

    #get commandline arguments
    args = cmdparser()
    
    #if cpu mode, hide all gpus available on the system
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    

    #tf session configuration
    print(f"\n#Jeager 0.01v will use the following parameters")
    print(f"input file : {args.input}")
    print(f"fragment size: {2048}")
    print(f"stride: {args.stride}")
    print(f"mode: meta\n{'-'*100}")
    
    gpus=get_gpu_count()
    if gpus > 0:
        print(f"gpus detected: {2}")
        
    else:
        print("We could not detect any gpu on this system.\nfor optimal performance run Jaeger on a GPU.")
    
    
    stratergy = tf.distribute.MirroredStrategy()
    with stratergy.scope():
    #build model and load weights
        inputs, outputs = LSTM_model(input_shape=(2048,4))
        model = CustomModel(inputs=inputs, outputs=outputs)
        model.load_weights(filepath='./weights/Mulitclass-5-LSTM.h5')#.expect_partial() when loading weights from a chpt file
        print(f"initialiting model and loading weights \u2705\n{'-'*100}")

        #run prediction loop 
        #open 2 files handles within the contex manager
        #with open(args.input) as fh, \
        fasta_file = args.input



        fh=get_compressed_file_handle(fasta_file)
        with open(args.output, 'w') as wfh:#, open(fasta_file) as fh:
            #header line
            wfh.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format('contig_id','length','#num_prok_windows','#num_vir_windows','#num_fun_windows',
                                                                                        '#num_prot_windows','#num_arch_windows','prediction','bac_score',
                                                                                        'vir_score', 'fun_score', 'prot_score', 'arch_score', 'window_summary'))
            num=0
            c=0
            c3 = [0,0,0,0,0]
            for i in fh: 
                if i.startswith('>'):
                    num+=1
            fh.seek(0)
            print(f'reading input and loading the model \u2705 \n{"-"*100}')
            
            prcs=process_input_string(table=mapper(),onehot=True)

            input_dataset = tf.data.Dataset.from_generator(
                     fasta_gen(fh,fragsize=2048,stride=args.stride,num=num),
                     output_signature=(
                         tf.TensorSpec(shape=(), dtype=tf.string)))

            idataset=input_dataset.map(prcs,
                                    num_parallel_calls=tf.data.AUTOTUNE).padded_batch(args.batch,
                                                                                      padded_shapes=((2048,4),(),(),(),(),()))
            for a,s,d,f,l in ExtractPredPEntry(idataset):
                    c+=1

                    #code for further processing goes here
                    #print(np.array(a), PerClassPreds(s))
                    c1 = PerClassPreds(s)
                    c4=AveragePerClassScore(a)
                    c3[np.argmax(c1)]+=1
                    c2=GetClass(c4, args.getalllabels)
                    c5=pred2string(s)
                    #print(c4)
                    wfh.write(f"{d[-1].decode()}\t{l[-1]}\t{c1[0]}\t{c1[1]}\t{c1[2]}\t{c1[3]}\t{c1[4]}\t{c2[0]}\t{c4[0]:.4f}\t{c4[1]:.4f}\t{c4[2]:.4f}\t{c4[3]:.4f}\t{c4[4]:.4f}\t{c5}\n")

        print(f"{'-'*100}\nprocessed {c} sequences 🦝" )
        fh.close()
                     #   log.write(f'skipping {record.id} because its shorter.length : {len(record.seq)}\n')
