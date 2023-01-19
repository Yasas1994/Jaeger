'''/**
 * Jaeger project (AA) v2.0.0- Identification of viruses in metagenomes 
                    using deep learning
 * File: inference.py; inference programs for the karas model
 * User: Y. Wijesekara
 * Date: 1/6/22
 * Time: 12:54 PM
 * Email: yasas.wijesekara@uni-greifswald.de
 */'''


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from nnlib import *
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
def signal_fl(it):
    '''get a signal at the begining and the end of a iterator'''
    iterable = iter(it)
    yield True, next(iterable)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var
def signal_l(it):
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

def get_num_entries(file_handle):
  '''returns the number of entries in a fasta file'''
  pass

def is_low_quality(sequence):
  '''returns True if more than 30% of the input sequence/fragment is covered with ambigious nucleotides 
     predictions made on such regions will be masked in the final prediction
  '''
  pass 

###################input pre-processing#################################
#map codons to amino acids

def codon_mapper():
    trimers=tf.constant(['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG',
            'ATT','ATC','ATA','ATG','GTT','GTC','GTA','GTG',
            'TCT','TCC','TCA','TCG','CCT','CCC','CCA','CCG',
            'ACT','ACC','ACA','ACG','GCT','GCC','GCA','GCG',
            'TAT','TAC','TAA','TAG','CAT','CAC','CAA','CAG',
            'AAT','AAC','AAA','AAG','GAT','GAC','GAA','GAG',
            'TGT','TGC','TGA','TGG','CGT','CGC','CGA','CGG',
            'AGT','AGC','AGA','AGG','GGT','GGC','GGA','GGG'])
    trimer_vals = tf.constant([1,1,2,2,2,2,2,2,3,3,3,4,5,5,5,5,6,6,6,6,
7,7,7,7,8,8,8,8,9,9,9,9,10,10,11,11,12,12,13,13,14,
14,15,15,16,16,17,17,18,18,11,19,20,20,20,20,6,6,20,
20,21,21,21,21
])
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    trimer_table = tf.lookup.StaticHashTable(trimer_init, default_value=0)
    
    return trimer_table
#convert to complement 
def c_mapper():
    rc_keys = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
    rc_vals = tf.constant([b'T', b'A', b'C', b'G',b'T', b'A', b'C', b'G'])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    rc_table = tf.lookup.StaticHashTable(rc_init, default_value="N")
    
    return rc_table

#new functions
def fasta_gen(filehandle,fragsize=None,stride=None,num=None): #fasta sequence generator
    '''returns a nucleotide fragment generator '''
    #filename here is a reference to a file handle
    #should also be able to handle small sequences
    #simply remove short sequnces, write to a separate file 
    def c():
        #accepts a reference to a file handle
        
            for record in tqdm(SeqIO.FastaIO.SimpleFastaParser(filehandle), total=num, bar_format='{l_bar}{bar:80}{r_bar}{bar:-10b}'):
                seqlen=len(record[1]) #move size filtering to a separate preprocessing step
                if seqlen >= fragsize: #filters the sequence based on size
                    if fragsize is None: #if no fragsize, return the entire sequence 
                        yield str(record[1])+","+str(record[0]) #sequence and sequence headder 
                    elif fragsize is not None:
                        for i,(l,index) in enumerate(signal_l(range(0,seqlen-(fragsize-1),fragsize if stride is None else stride))):
                            yield str(record[1])[index:index+fragsize]+","+str(record[0].replace(',',''))+","+str(index)+","+str(l)+","+str(i)+","+str(seqlen)
    return c #return fasta sequence generator

@tf.function
def process_string(string, t1=codon_mapper(), t3=c_mapper(),onehot=True, label_onehot=True):

    x = tf.strings.split(string, sep=',')
    

    forward_strand = tf.strings.bytes_split(x[0])#split the string 
    reverse_strand = t3.lookup(forward_strand[::-1])
    

    tri_forward =tf.strings.ngrams(forward_strand,ngram_width=3,separator='')
    tri_reverse =tf.strings.ngrams(reverse_strand,ngram_width=3,separator='')
    
    f1=t1.lookup(tri_forward[::3])
    f2=t1.lookup(tri_forward[1::3])
    f3=t1.lookup(tri_forward[2::3])
    
    r1=t1.lookup(tri_reverse[::3])
    r2=t1.lookup(tri_reverse[1::3])
    r3=t1.lookup(tri_reverse[2::3])
    

    return {"forward_1": f1, "forward_2": f2, "forward_3": f3, "reverse_1": r1, "reverse_2" : r2, "reverse_3" : r3 }, x[1], x[2], x[3], x[4], x[5]


######################################calling the model########################################
def get_predictions(idataset, model):  #get predictions per batch  

    for batch in idataset:
        probs= tf.keras.activations.softmax(model(batch[0])) #convert logits to probabilities 
        #probsperpos.append(probs) #probabilities per position
        y_pred= tf.argmax(probs,-1) #get predicted class for each instance in the batch   
        yield probs, y_pred, batch[1], batch[2], batch[3], batch[4], batch[5]


def extract_pred_entry(model,idataset, numclass=4):#takes a generator as input 
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
    
    for prob,y_pred,id_,pos_,is_last_,index_,clen_ in get_predictions(idataset,model):
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
                

    
def per_class_preds(y_pred): #y_pred is a vector with scores for the entire entry
    return np.array([tf.reduce_sum(tf.cast(y_pred==i, tf.float32)).numpy() 
                                                  for i in range(4)])

def average_per_class_score(yprob):
    return np.mean(yprob, axis=0)
    
def per_class_preds(y_pred,numclass=4): #y_pred is a vector with scores for the entire entry
    return np.array([np.sum(y_pred==i) for i in range(numclass)])

def average_per_class_score(yprob):
    return np.mean(yprob, axis=0)

def get_class(y_pred, get_all_classes=False): 
    '''Protista was removed from v2.0.0 '''
    y_pred=y_pred/sum(y_pred)
    c=np.argmax(y_pred)
    if c == 0:
        return "Prokaryota" if get_all_classes else "Non-phage",round(y_pred[c],3)
    elif c == 1:
        return "Phage",round(y_pred[c],3)
    elif c == 2:
        return "Fungi" if get_all_classes else "Non-phage",round(y_pred[c],3)
    elif c==3:
        return "Archaea" if get_all_classes else "Non-phage",round(y_pred[c],3)
    else:
        raise "class can not be >4"

def get_hs_positions(probsperpos, numclass=4):    
    '''extracts high scoring positions from each class
    return a dict'''
    y_preds=np.argmax(probsperpos,axis=1)
    #get high scoring positions for each class
    scorepos={class_:np.where(np.argmax(probsperpos,axis=1)==class_)[0] for class_ in range(numclass)}
    return scorepos

def get_perclass_probs(probperpos, numclass=4):
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
def per_class_preds(y_pred): #y_pred is a vector with scores for the entire entry
    return np.array([np.sum(y_pred==i) for i in range(4)])


################################get records with matching ids###################################

def write_to_file(input_fasta_fh,output_table,ofasta_filename, cut_off=0.6, len_cut = 2048):
    '''writes predicted viral contigs with scores > a cut-off to a new file'''
    list_of_contig_id={}
    with open(output_table, 'r') as fh:
        for i,line in enumerate(fh):
            if i > 0:
                cols=line.split('\t')
                if (float(cols[8]) > cut_off) and (int(cols[1]) >= len_cut): # if viral score is greater than the cut-off 
                    list_of_contig_id[cols[0]] = 0

    with open(ofasta_filename, 'w') as fh:
        for record in SeqIO.FastaIO.SimpleFastaParser(input_fasta_fh):
            if str(record[0].replace(',','')) in list_of_contig_id:
                fh.write(f'>{record[0]}\n{record[1]}\n')

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
    parser = argparse.ArgumentParser(description='\n## Jaeger_AA : Incooperating deep learning into phage discovery https://github.com/Yasas1994/Jaeger.git',
    usage=argparse.SUPPRESS,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i","--input",
                        type=file_path,
                        required=True,
                        help="path to input file")
    parser.add_argument("-o","--output", 
                        type=str,
                        required=True,
                        help='path to output file')
    parser.add_argument("-of","--ofasta", 
                        type=str,
                        required=False,
                        help='path to output fasta file')
    parser.add_argument("--cutoff", 
                        type=float,
                        required=False,
                        default=0.75,
                        help='fasta output cutoff score')
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
    
    seed = 8959
    tf.random.set_seed(seed)

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
        tf.config.set_visible_devices([], 'GPU:1')
    

    #tf session configuration
    print(f"\n#Jeager v2.0.0 will use the following parameters")
    print(f"input file : {args.input}")
    print(f"fragment size: {2048}")
    print(f"stride: {args.stride}")
    print(f"batch size: {args.batch}")
    print(f"mode: meta\n{'-'*100}")
    
    gpus=get_gpu_count()
    if gpus > 0:
        print(f"gpus detected: {gpus}")
        
    else:
        print("We could not detect any gpu on this system.\nfor optimal performance run Jaeger on a GPU.")
    
    fasta_file = args.input
    fhi=get_compressed_file_handle(fasta_file)    
    stratergy = tf.distribute.MirroredStrategy()
    #generate predictions and write to the table
    with stratergy.scope():
    #build model and load weights
        inputs, outputs = LSTM_model(input_shape=(None,))
        model = CustomModel(inputs=inputs, outputs=outputs)
        model.load_weights(filepath='./weights/lstm_codon_aa.h5')#.expect_partial() when loading weights from a chpt file
        print(f"initialiting model and loading weights \u2705\n{'-'*100}")
    
        #run prediction loop 
        #open 2 files handles within the contex manager
        #with open(args.input) as fh, \


        with open(args.output, 'w') as wfh:#, open(fasta_file) as fh:
            #header line
            wfh.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format('contig_id','length','#num_prok_windows','#num_vir_windows','#num_fun_windows',
                                                                                        '#num_arch_windows','prediction','bac_score',
                                                                                        'vir_score', 'fun_score','arch_score', 'window_summary'))
            num=0
            c=0
            c3 = [0,0,0,0]
            for i in fhi: 
                if i.startswith('>'):
                    num+=1
            fhi.seek(0)
            print(f'reading input and loading the model \u2705 \n{"-"*100}')
            
            #prcs=process_input_string(table=mapper(),onehot=True)
            
            with tf.device('/cpu:0'):
                input_dataset = tf.data.Dataset.from_generator( fasta_gen(fhi,fragsize=2048,stride=args.stride,num=num),output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)))

                idataset=input_dataset.map(process_string,num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch, num_parallel_calls=tf.data.AUTOTUNE)
            for a,s,d,f,l in extract_pred_entry(model,idataset):
                    c+=1

                    #code for further processing goes here
                    #print(np.array(a), PerClassPreds(s))
                    c1 = per_class_preds(s)
                    c4=average_per_class_score(a)
                    c3[np.argmax(c1)]+=1
                    c2=get_class(c4, args.getalllabels)
                    c5=pred2string(s)
                    #print(c4)
                    wfh.write(f"{d[-1].decode()}\t{l[-1]}\t{c1[0]}\t{c1[1]}\t{c1[2]}\t{c1[3]}\t{c2[0]}\t{c4[0]:.4f}\t{c4[1]:.4f}\t{c4[2]:.4f}\t{c4[3]:.4f}\t{c5}\n")

        print(f"{'-'*100}\nprocessed {c} sequences 🦝" )
    if args.ofasta:
        fhi.seek(0)
        print("dumping predicted contigs to a fasta file\n")
        write_to_file(input_fasta_fh=fhi,output_table=args.output,ofasta_filename=args.ofasta, cut_off=args.cutoff)
        
    fhi.close()

                     #   log.write(f'skipping {record.id} because its shorter.length : {len(record.seq)}\n')