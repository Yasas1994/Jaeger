import tensorflow as tf
from preprocess.latest.maps import *

#map codons to amino acids
def codon_mapper():
    trimers=tf.constant(codons)
    trimer_vals = tf.constant(codon_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    trimer_table = tf.lookup.StaticHashTable(trimer_init, default_value=-1)
    
    return trimer_table

def codon_bias_mapper():
    trimers=tf.constant(codons)
    trimer_vals = tf.constant(cod_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    trimer_table = tf.lookup.StaticHashTable(trimer_init, default_value=-1)
    
    return trimer_table

#convert to complement 
def c_mapper():
    rc_keys = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
    rc_vals = tf.constant([b'T', b'A', b'C', b'G',b't', b'a', b'c', b'g'])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    rc_table = tf.lookup.StaticHashTable(rc_init, default_value="N")
    
    return rc_table

    
def mapper2():
    keys_tensor = tf.constant([0,1,2,3,4,5,6])
    vals_tensor = tf.constant([0,1,2,3,4,5,6]) #[0,1,2,3,4,5,6]
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    t2 = tf.lookup.StaticHashTable(ini, default_value=0)
    return t2

def mapper3():
    keys_tensor = tf.constant([b'A', b'G', b'C', b'T' ,b'a', b'g', b'c', b't'])
    vals_tensor = tf.constant([0, 1, 2, 3, 0, 1, 2, 3])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    t2 = tf.lookup.StaticHashTable(ini, default_value=-1)
    return t2

def mapper4():
    keys_tensor = tf.constant([b'A', b'G', b'C', b'T'])
    vals_tensor = tf.constant([0,1,1,0])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    t2 = tf.lookup.StaticHashTable(ini, default_value=-1)
    return t2

def process_string_train(t1=codon_mapper(),t2=mapper2(), t3=c_mapper(),t4=mapper3(),t5=codon_bias_mapper(),
                    class_label_onehot=True,
                    label_type="classifier", #or "reliability"
                    num_classes=5,
                    crop_size=1024,
                    timesteps=False,num_time = None, fragsize=200,
                    mutate = False, mutation_rate=0.1):
    @tf.function
    def p(string):
        x = tf.strings.split(string, sep=',')
        label= t2.lookup(tf.strings.to_number(x[0], tf.int32))
        label =tf.cast(label, dtype=tf.int32)



        #start = tf.random.uniform(shape = (1,),  minval=0, maxval=100, dtype=tf.int32)[0]
        if (crop_size%3) == 0:
            offset = -2   
        elif (crop_size%3) == 1:
            offset = -1
        elif (crop_size%3) == 2:
            offset = 0
        x_len = tf.strings.length(x[1])

    
        forward_strand = tf.strings.bytes_split(x[1])[0:crop_size]#split the string 
        if mutate:
            #int_array = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)  # or tf.Variable()
            mutation_prob = mutation_rate  # Probability of mutation (adjust as needed)
            min_value = 0       # Minimum possible value for mutation
            max_value = 4      # Maximum possible value for mutation

            alphabet = tf.constant(['A', 'T', 'G', 'C','N'], dtype=tf.string)
            
            mask = tf.random.uniform(shape=tf.shape(forward_strand), minval=0.0, maxval=1.0) < mutation_prob
            mutation_values = tf.random.uniform(shape=tf.shape(forward_strand), minval=min_value, maxval=max_value, dtype=tf.int32)
            selected_strings = tf.gather(alphabet, mutation_values)
            forward_strand = tf.where(mask, selected_strings, forward_strand)
            
        reverse_strand = t3.lookup(forward_strand[::-1]) #reverse complement
        
        nuc1 = t4.lookup(forward_strand[:])
        nuc2 = t4.lookup(reverse_strand[:])
        
        tri_forward = tf.strings.ngrams(forward_strand, ngram_width=3,separator='')
        tri_reverse = tf.strings.ngrams(reverse_strand, ngram_width=3,separator='')
        
        # if x_len < crop_size+3:
        #     y = tf.constant([0],dtype=tf.int32)
        #     pad_size = tf.math.maximum(num_time*fragsize - tf.size(tri_forward[0:-3:3]),y)
        #     paddings = [[0, pad_size]] # assign here, during graph execution
        # else:
        #     paddings = [[0, 0]] 

#         f1=tf.pad(t1.lookup(tri_forward[0:-3:3]), paddings)
#         f2=tf.pad(t1.lookup(tri_forward[1:-2:3]), paddings)
#         f3=tf.pad(t1.lookup(tri_forward[2:-1:3]), paddings)
        
#         fb1=tf.pad(t5.lookup(tri_forward[0:-3:3]), paddings)
#         fb2=tf.pad(t5.lookup(tri_forward[1:-2:3]), paddings)
#         fb3=tf.pad(t5.lookup(tri_forward[2:-1:3]), paddings)

#         r1=tf.pad(t1.lookup(tri_reverse[0:-3:3]), paddings)
#         r2=tf.pad(t1.lookup(tri_reverse[1:-2:3]), paddings)
#         r3=tf.pad(t1.lookup(tri_reverse[2:-1:3]), paddings)
        
#         rb1=tf.pad(t5.lookup(tri_reverse[0:-3:3]), paddings)
#         rb2=tf.pad(t5.lookup(tri_reverse[1:-2:3]), paddings)
#         rb3=tf.pad(t5.lookup(tri_reverse[2:-1:3]), paddings)
        
        f1=t1.lookup(tri_forward[0:-3+offset:3])
        f2=t1.lookup(tri_forward[1:-2+offset:3])
        f3=t1.lookup(tri_forward[2:-1+offset:3])
        
        fb1=t5.lookup(tri_forward[0:-3+offset:3])
        fb2=t5.lookup(tri_forward[1:-2+offset:3])
        fb3=t5.lookup(tri_forward[2:-1+offset:3])

        r1=t1.lookup(tri_reverse[0:-3+offset:3])
        r2=t1.lookup(tri_reverse[1:-2+offset:3])
        r3=t1.lookup(tri_reverse[2:-1+offset:3])
        
        rb1=t5.lookup(tri_reverse[0:-3+offset:3])
        rb2=t5.lookup(tri_reverse[1:-2+offset:3])
        rb3=t5.lookup(tri_reverse[2:-1+offset:3])
        #tf.print(tf.size(r1),tf.size(r2),tf.size(r3),tf.size(f1),tf.size(f2),tf.size(f3),x_len)
       
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
            nuc = tf.stack([nuc1,nuc2],0)
            code = tf.stack([fb1,fb2,fb3,rb1,rb2,rb3],0)
            

        if label_type == "reliability":
                reliability = tf.expand_dims(label, -1)
                if class_label_onehot:
                    label = tf.one_hot(label, depth=num_classes, dtype=tf.float32, on_value=1, off_value=0)
                return {'translated':tf.one_hot(seq, depth=64, dtype=tf.float32, on_value=1, off_value=0),
                        'nucleotide': tf.one_hot(nuc, depth=4, dtype=tf.float32, on_value=1, off_value=0)}, \
                       {'classifier': label, 'reliability': reliability}
            
        elif label_type == "classifier":
                reliability_dummy = tf.expand_dims(tf.zeros_like(label), -1)
                if class_label_onehot:
                    label = tf.one_hot(label, depth=num_classes, dtype=tf.float32, on_value=1, off_value=0)
                return {'translated':tf.one_hot(seq, depth=64, dtype=tf.float32, on_value=1, off_value=0),
                        'nucleotide': tf.one_hot(nuc, depth=4, dtype=tf.float32, on_value=1, off_value=0)}, \
                       {'classifier': label, 'reliability': reliability_dummy}
        else:
                raise ValueError(f"{label} is not a invalid [classifier or reliability]")
    
        #tf.one_hot(nuc, depth=4, dtype=tf.float32, on_value=1, off_value=0)

    return p

