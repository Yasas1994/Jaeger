import numpy as np
from Bio import SeqIO
import tensorflow as tf

def get_predictions(idataset, model):  #get predictions per batch  

    for batch in idataset:
        logits = model(batch[0]).numpy()
        probs= tf.nn.softmax(logits) #convert logits to probabilities 
        #probsperpos.append(probs) #probabilities per position
        y_pred= np.argmax(probs,-1) #get predicted class for each instance in the batch   
        yield logits , y_pred, batch[1], batch[2], batch[3], batch[4], batch[5]

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
    #y_pred=y_pred/sum(y_pred)
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
