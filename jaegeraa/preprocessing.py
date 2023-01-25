import tensorflow as tf
from tqdm import tqdm
from Bio import SeqIO, Seq
from jaegeraa.utils import signal_fl, signal_l
import io
import types

def codon_mapper():
    '''hash table convert codons to corresponding aminoacid.
       each amino acid is represented by an integer.
    '''
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

def c_mapper():
    '''returns a hash table that maps the input nucleotide string to its complementry sequence'''
    rc_keys = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
    rc_vals = tf.constant([b'T', b'A', b'C', b'G',b'T', b'A', b'C', b'G'])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    rc_table = tf.lookup.StaticHashTable(rc_init, default_value="N")
    
    return rc_table

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
    return c


def fasta_gen_lib(filehandle,fragsize=None,stride=None,num=None): #fasta sequence generator
    '''returns a nucleotide fragment generator '''
    #filename here is a reference to a file handle
    #should also be able to handle small sequences
    #simply remove short sequnces, write to a separate file 
    head = False
    if isinstance(filehandle,str):
        #checks if input is a string
        tmpfn = [filehandle]
    elif isinstance(filehandle,list):
        #checks if input is a list
        tmpfn = filehandle
        if len(tmpfn[-1]) == 2:
            head = True

    elif isinstance(filehandle,Seq):
        #checks if the input is a Seq object
        tmpfn = [str(filehandle)]
        
    elif isinstance(filehandle,types.GeneratorType):
        #checks if the input is a generator
        tmpfn = filehandle
        
    elif isinstance(filehandle,io.TextIOWrapper):
        #checks if the input is a fileobject
        tmpfn = SeqIO.FastaIO.SimpleFastaParser(filehandle)
        head = True
    else:
        raise ValueError('Not a supported input type')
    def c():
        #accepts a reference to a file handle
            for n,record in enumerate(tmpfn):
                if head:
                    seqlen=len(record[1]) #move size filtering to a separate preprocessing step
                    seq = record[1]
                    headder = record[0]
                else:
                    seqlen = len(record)
                    seq = record
                    headder = f'seq_{n}'
                if seqlen >= fragsize: #filters the sequence based on size
                    if fragsize is None: #if no fragsize, return the entire sequence 
                        yield str(seq)+","+str(headder) #sequence and sequence headder 
                    elif fragsize is not None:
                        for i,(l,index) in enumerate(signal_l(range(0,seqlen-(fragsize-1),fragsize if stride is None else stride))):
                            yield str(seq)[index:index+fragsize]+","+str(headder.replace(',',''))+","+str(index)+","+str(l)+","+str(i)+","+str(seqlen)
    return c

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

