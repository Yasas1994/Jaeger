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
def amino_mapper():
    '''maps each amino acid to a integer'''
    aa = tf.constant(['F','L','I','M','V',
                      'S','P','T','A','Y',
                      '*','H','Q','N','K',
                      'D','E','C','W',
                      'R','G',])
    aa_num = tf.constant([1,2,3,4,5,
                          6,7,8,9,10,
                          21,11,12,13,
                          14,15,16,17,
                          18,19,20])
    aa_init = tf.lookup.KeyValueTensorInitializer(aa, aa_num)
    aa_table = tf.lookup.StaticHashTable(aa_init, default_value=0)
    
    return aa_table

def c_mapper():
    '''returns a hash table that maps the input nucleotide string to its complementry sequence'''
    rc_keys = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
    rc_vals = tf.constant([b'T', b'A', b'C', b'G',b't', b'a', b'c', b'g'])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    rc_table = tf.lookup.StaticHashTable(rc_init, default_value="N")
    
    return rc_table

def fasta_gen(filehandle,fragsize=None,stride=None,num=None, disable=False): #fasta sequence generator
    '''returns a nucleotide fragment generator '''
    #filename here is a reference to a file handle
    #should also be able to handle small sequences
    #simply remove short sequnces, write to a separate file 
    def c():
        #accepts a reference to a file handle
        
            for record in tqdm(SeqIO.FastaIO.SimpleFastaParser(filehandle), total=num, ascii=' >=',bar_format='{l_bar}{bar:10}{r_bar}',dynamic_ncols=True,unit='seq' ,colour='green', disable=disable):
            #for record in SeqIO.FastaIO.SimpleFastaParser(filehandle):
                seqlen=len(record[1]) #move size filtering to a separate preprocessing step
                sequence = str(record[1]).upper()
                if seqlen >= fragsize: #filters the sequence based on size
                    if fragsize is None: #if no fragsize, return the entire sequence 
                        yield sequence +","+str(record[0].replace(',','_')) #sequence and sequence headder 
                    elif fragsize is not None:
                        
                        for i,(l,index) in enumerate(signal_l(range(0,seqlen-(fragsize-1),fragsize if stride is None else stride))):
                            yield sequence[index:index+fragsize]+","+str(record[0].split(',')[0])+","+str(index)+","+str(l)+","+str(i)+","+str(seqlen)
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
                    headder = record[0].replace(',','__')
             
                else:
                    seqlen = len(record)
                    seq = record
                    headder = f'seq_{n}'
                if seqlen >= fragsize: #filters the sequence based on size
                    if fragsize is None: #if no fragsize, return the entire sequence 
                        yield str(seq)+","+str(headder) #sequence and sequence headder 
                    elif fragsize is not None:
                        for i,(l,index) in enumerate(signal_l(range(0,seqlen-(fragsize-1),fragsize if stride is None else stride))):
                            yield str(seq)[index:index+fragsize]+","+str(headder)+","+str(index)+","+str(l)+","+str(i)+","+str(seqlen)
    return c


def process_string(onehot=True, label_onehot=True, crop_size=2048):
    @tf.function
    def p(string):
        t1,t3=codon_mapper(), c_mapper()
        x = tf.strings.split(string, sep=',')
        
        if (crop_size%3) == 0:
            offset = -2   
        elif (crop_size%3) == 1:
            offset = -1
        elif (crop_size%3) == 2:
            offset = 0

        forward_strand = tf.strings.bytes_split(x[0])#split the string 
        reverse_strand = t3.lookup(forward_strand[::-1])
        

        tri_forward =tf.strings.ngrams(forward_strand,ngram_width=3,separator='')
        tri_reverse =tf.strings.ngrams(reverse_strand,ngram_width=3,separator='')
        
        f1=t1.lookup(tri_forward[0:-3+offset:3])
        f2=t1.lookup(tri_forward[1:-2+offset:3])
        f3=t1.lookup(tri_forward[2:-1+offset:3])
        
        r1=t1.lookup(tri_reverse[0:-3+offset:3])
        r2=t1.lookup(tri_reverse[1:-2+offset:3])
        r3=t1.lookup(tri_reverse[2:-1+offset:3])
        

        return {"forward_1": f1, "forward_2": f2, "forward_3": f3, "reverse_1": r1, "reverse_2" : r2, "reverse_3" : r3 }, x[1], x[2], x[3], x[4], x[5]
    return p

def process_string_textline_protein(label_onehot=True,numclasses=4):
    
    def p(string):
        t1=amino_mapper()

        x = tf.strings.split(string, sep=',')

        label= tf.strings.to_number(x[0], tf.int32)
        label= tf.cast(label, dtype=tf.int32)

        prot_strand = tf.strings.bytes_split(x[1])#split the string 

        protein=t1.lookup(prot_strand)

        if label_onehot:
            label = tf.one_hot(label, depth=numclasses, dtype=tf.float32, on_value=1, off_value=0)

        return protein, label
    return p

def process_string_textline(onehot=True, label_onehot=True,numclasses=4):
    
    def p(string):
        t1,t3=codon_mapper(), c_mapper()

        x = tf.strings.split(string, sep=',')

        label= tf.strings.to_number(x[0], tf.int32)
        label= tf.cast(label, dtype=tf.int32)


            
        forward_strand = tf.strings.bytes_split(x[1])#split the string 
        reverse_strand = t3.lookup(forward_strand[::-1])


        tri_forward = tf.strings.ngrams(forward_strand,ngram_width=3,separator='')
        tri_reverse = tf.strings.ngrams(reverse_strand,ngram_width=3,separator='')

        f1=t1.lookup(tri_forward[::3])
        f2=t1.lookup(tri_forward[1::3])
        f3=t1.lookup(tri_forward[2::3])

        r1=t1.lookup(tri_reverse[::3])
        r2=t1.lookup(tri_reverse[1::3])
        r3=t1.lookup(tri_reverse[2::3])


        if label_onehot:
            label = tf.one_hot(label, depth=numclasses, dtype=tf.float32, on_value=1, off_value=0)

        return {"forward_1": f1, "forward_2": f2, "forward_3": f3, "reverse_1": r1, "reverse_2" : r2, "reverse_3" : r3 }, label
    return p

#Second generation preprocessing code
codons  = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG',
            'ATT','ATC','ATA','ATG','GTT','GTC','GTA','GTG',
            'TCT','TCC','TCA','TCG','CCT','CCC','CCA','CCG',
            'ACT','ACC','ACA','ACG','GCT','GCC','GCA','GCG',
            'TAT','TAC','TAA','TAG','CAT','CAC','CAA','CAG',
            'AAT','AAC','AAA','AAG','GAT','GAC','GAA','GAG',
            'TGT','TGC','TGA','TGG','CGT','CGC','CGA','CGG',
            'AGT','AGC','AGA','AGG','GGT','GGC','GGA','GGG']

aa = ['F','F','L','L','L','L','L','L',
    'I','I','I','M','V','V','V','V',
    'S','S','S','S','P','P','P','P',
    'T','T','T','T','A','A','A','A',
    'Y','Y','*','*','H','H','Q','Q',
    'N','N','K','K','D','D','E','E',
    'C','C','*','W','R','R','R','R',
    'S','S','R','R','G','G','G','G',]



pc2 = {"I":"A","V":"A","L":"A","F":"A","Y":"A","W":"A","H":"B","K":"B","R":"B","D":"B",
"E":"B","G":"A","A":"A","C":"A","S":"A", "T":"A","M":"A","Q":"B","N":"B","P":"A"}

pc2_num =[1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,0,0,2,2,2,2,
        2,2,2,2,2,2,2,2,
        1,1,0,1,2,2,2,2,
        1,1,2,2,1,1,1,1,]

murphy10={
"A":"A","C":"C","G":"G","H":"H","P":"P","L":"L","V":"L","I":"L","M":"L","S":"S",
"T":"S","F":"F","Y":"F","W":"F","E":"E","D":"E","N":"E","Q":"E","K":"K","R":"K"}

pc5={
"I":"A","V":"A","L":"A", "F":"R","Y":"R","W":"R","H":"R","K":"C","R":"C","D":"C",
"E":"C","G":"T","A":"T","C":"T","S":"T", "T":"D","M":"D","Q":"D","N":"D","P":"D"
}

aa_num = [1,1,2,2,2,2,2,2,
          3,3,3,4,5,5,5,5,
          6,6,6,6,7,7,7,7,
          8,8,8,8,9,9,9,9,
          10,10,0,0,11,11,12,12,
          13,13,14,14,15,15,16,16,
          17,17,0,18,19,19,19,19,
          6,6,19,19,20,20,20,20
]
cod_num = [1,2,1,2,3,4,5,6,
           1,2,3,1,1,2,3,4,
           1,2,3,4,1,2,3,4,
           1,2,3,4,1,2,3,4,
           1,2,1,2,1,2,1,2,
           1,2,1,2,1,2,1,2,
           1,2,1,2,1,2,3,4,
           5,6,5,6,1,2,3,4
]

murphy10_num = [1,1,2,2,2,2,2,2,
                2,2,2,2,2,2,2,2,
                3,3,3,3,4,4,4,4,
                3,3,3,3,5,5,5,5,
                1,1,0,0,6,6,7,7,
                7,7,8,8,7,7,7,7,
                9,9,0,1,8,8,8,8,
                3,3,8,8,10,10,10,10]

pc5_num = [1,1,2,2,2,2,2,2,
           2,2,2,3,2,2,2,2,
           4,4,4,4,3,3,3,3,
           3,3,3,3,4,4,4,4,
           1,1,0,0,1,1,3,3,
           3,3,5,5,5,5,5,5,
           4,4,0,1,5,5,5,5,
           4,4,5,5,4,4,4,4]

codon_num = [i for i in range(0,65)]


#map codons to amino acids
def codon_mapper_gen2():
    trimers=tf.constant(codons)
    trimer_vals = tf.constant(murphy10_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    trimer_table = tf.lookup.StaticHashTable(trimer_init, default_value=0)
    
    return trimer_table

def codon_bias_mapper():
    trimers=tf.constant(codons)
    trimer_vals = tf.constant(cod_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    trimer_table = tf.lookup.StaticHashTable(trimer_init, default_value=0)
    
    return trimer_table

#convert to complement 
def complement_mapper():
    rc_keys = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
    rc_vals = tf.constant([b'T', b'A', b'C', b'G',b't', b'a', b'c', b'g'])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    rc_table = tf.lookup.StaticHashTable(rc_init, default_value="N")
    
    return rc_table

def nuc_enc_mapper():
    keys_tensor = tf.constant([b'A', b'G', b'C', b'T',b'a', b'g', b'c', b't'])
    vals_tensor = tf.constant([0,1,2,3,0,1,2,3])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    nem = tf.lookup.StaticHashTable(ini, default_value=-1)
    return nem

def alt_nuc_enc_mapper():
    keys_tensor = tf.constant([b'A', b'G', b'C', b'T',b'a', b'g', b'c', b't'])
    vals_tensor = tf.constant([0,1,1,0,0,1,1,0])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    anem = tf.lookup.StaticHashTable(ini, default_value=-1)
    return anem

def process_string_gen2(onehot=True,
                        crop_size=1024,
                        maxval=400,
                        timesteps=False,
                        num_time = None,
                        fragsize=200,
                        mutate = False, 
                        mutation_rate=0.1):
    @tf.function
    def p(string):
        
        t1=codon_mapper_gen2()
        t3=complement_mapper()
        t4=nuc_enc_mapper()
        t5=codon_bias_mapper()
        
        x = tf.strings.split(string, sep=',')

        if (crop_size%3) == 0:
            offset = -2   
        elif (crop_size%3) == 1:
            offset = -1
        elif (crop_size%3) == 2:
            offset = 0
        x_len = tf.strings.length(x[0])

    
        forward_strand = tf.strings.bytes_split(x[0])[0:crop_size]
        
        if mutate:

            mutation_prob = mutation_rate  # Probability of mutation (adjust as needed)
            min_value = 0       # Minimum possible value for mutation
            max_value = 4      # Maximum possible value for mutation

            alphabet = tf.constant(['A', 'T', 'G', 'C','N'], dtype=tf.string)
            
            mask = tf.random.uniform(shape=tf.shape(forward_strand), minval=0.0, maxval=1.0) < mutation_prob
            mutation_values = tf.random.uniform(shape=tf.shape(forward_strand), minval=min_value, maxval=max_value, dtype=tf.int32)
            selected_strings = tf.gather(alphabet, mutation_values)
            forward_strand = tf.where(mask, selected_strings, forward_strand)
            
        #generate the reverse strand for the mutated forward strand
        reverse_strand = t3.lookup(forward_strand[::-1])
        
        nuc1 = t4.lookup(forward_strand[:])
        nuc2 = t4.lookup(reverse_strand[:])
        
        tri_forward = tf.strings.ngrams(forward_strand,ngram_width=3,separator='')
        tri_reverse = tf.strings.ngrams(reverse_strand,ngram_width=3,separator='')
        
        
        f1=t1.lookup(tri_forward[0:-3+offset:3])
        f2=t1.lookup(tri_forward[1:-2+offset:3])
        f3=t1.lookup(tri_forward[2:-1+offset:3])
        
        # fb1=t5.lookup(tri_forward[0:-3+offset:3])
        # fb2=t5.lookup(tri_forward[1:-2+offset:3])
        # fb3=t5.lookup(tri_forward[2:-1+offset:3])

        r1=t1.lookup(tri_reverse[0:-3+offset:3])
        r2=t1.lookup(tri_reverse[1:-2+offset:3])
        r3=t1.lookup(tri_reverse[2:-1+offset:3])
        
        # rb1=t5.lookup(tri_reverse[0:-3+offset:3])
        # rb2=t5.lookup(tri_reverse[1:-2+offset:3])
        # rb3=t5.lookup(tri_reverse[2:-1+offset:3])

       
        if timesteps:
            f1=tf.reshape(f1,(num_time,fragsize))
            f2=tf.reshape(f2,(num_time,fragsize))
            f3=tf.reshape(f3,(num_time,fragsize))
            r1=tf.reshape(r1,(num_time,fragsize))
            r2=tf.reshape(r2,(num_time,fragsize))
            r3=tf.reshape(r3,(num_time,fragsize))
            seq = tf.stack([f1,f2,f3,r1,r2,r3],1)
        else:
            seq = tf.stack([f1,f2,f3,r1,r2,r3],0)
            #nuc = tf.stack([nuc1,nuc2],0)
            # code = tf.stack([fb1,fb2,fb3,rb1,rb2,rb3],0) # codon bias encoder
            

        return {'translated':tf.one_hot(seq,depth=11, dtype=tf.float32, on_value=1, off_value=0)},x[1], x[2], x[3], x[4], x[5]
                #'nucleotide': tf.one_hot(nuc, depth=4, dtype=tf.float32, on_value=1, off_value=0)}, 
    return p
