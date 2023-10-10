import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import tensorflow as tf



def ood_predict(x_features, params):
    #use parameters extimated using sklearn
    x_features = (x_features- x_features.mean(axis=-1).reshape(-1,1))/x_features.std(axis=-1).reshape(-1,1)
    logits = np.dot(x_features,params['coeff'].reshape(-1,1)) + params['intercept']
    return (1/(1+ np.exp(logits))).flatten(), logits

def get_predictions(idataset, model, cutoffs, ood_params):  #get predictions per batch  

    for batch in idataset:
        logits = model(batch[0])
        if isinstance(logits,list):
            logits, ood_features = logits[0].numpy(), logits[1].numpy()
            if ood_params is not None:
                ood,_= ood_predict(ood_features, ood_params)
                
        else:
            logits = logits.numpy()
            ood = None
        probs= tf.nn.softmax(logits) #convert logits to probabilities 
        
        if cutoffs is not None:
            # if 2 or more above the cutoff use argmax
            
            
            above_cutoff = np.array(probs) >= cutoffs
            argmax = np.argmax(probs, axis=-1)
            above_cutoff_count = np.count_nonzero(above_cutoff, axis=-1)
            mask_cuttoff = above_cutoff_count == 1
            mask_argmax = above_cutoff_count > 1
            y_pred = argmax*mask_argmax + -1 #+ np.argmax(above_cutoff*mask_cuttoff.reshape(-1,1), axis=-1)
            # probs = probs*mask_argmax
            
        else:
            y_pred= np.argmax(probs,-1) #get predicted class for each instance in the batch   
    
        yield logits , y_pred, ood , batch[1], batch[2], batch[3], batch[4], batch[5]

def extract_pred_entry(model,idataset, numclass=4, cutoffs=None,ood_params=None):#takes a generator as input 
    '''
    #prob : position wise softmax prbs
    #y_pred : position wise predicted classes
    #id_: fasta headder
    #pos_:position in the sequence
    #index_:index at pos_[i]
    #is_last:end of sequence signal'''
    
    tmp_prob = np.empty((0,numclass),float) #empty np.array to store predicted probabilities
    tmp_ypred = np.empty((0),int) #empty np.array to store predicted classes
    tmp_ood = np.empty((0),float)
    tmp_id = [] #contig id 
    tmp_pos = [] #genomic cordinate vector -> used for prophage prediction 
    tmp_len = []
    
    for prob,y_pred,ood,id_,pos_,is_last_,index_,clen_ in get_predictions(idataset,model,cutoffs,ood_params):
        #
        #prb_tmp.extend([i.numpy() for i in prob])
        #print(prb_tmp)
        tmp_prob = np.append(tmp_prob, prob, axis=0) #concatenate the prob vectors 
        tmp_ypred = np.append(tmp_ypred, y_pred, axis=0) #concatenate the prob vectors 
        tmp_id.extend([i for i in id_.numpy()]) #adds contig ids
        tmp_pos.extend([int(i) for i in pos_.numpy()])
        tmp_len.extend([int(i) for i in clen_.numpy()])
        
        if ood is not None:
            tmp_ood = np.append(tmp_ood, ood, axis=0)
        else:
            tmp_ood = None
            tmp2 = None
            
        for index,is_last in zip(index_,is_last_): #when is_last signal is received, return the prob vector

            if int(is_last) == 1:
                #print(prb_tmp.shape)
                tmp = tmp_prob[:int(index)+1]
                tmp_prob = tmp_prob[int(index)+1:]

                tmp1 = tmp_ypred[:int(index)+1]
                tmp_ypred = tmp_ypred[int(index)+1:]
                
                if ood is not None:
                    tmp2   = tmp_ood[:int(index)+1]
                    tmp_ood = tmp_ood[int(index)+1:]

                tmp3   = tmp_id[:int(index)+1]
                tmp_id = tmp_id[int(index)+1:]

                tmp4   = tmp_pos[:int(index)+1]
                tmp_pos = tmp_pos[int(index)+1:]

                tmp5   = tmp_len[:int(index)+1]
                tmp_len = tmp_len[int(index)+1:]

                #print(prb_tmp.shape, tmp.shape)
                yield tmp, tmp1, tmp2, tmp3, tmp4, tmp5
                
def get_ood_probability(ood):
    if ood is not None:
        #print(ood)
        score = str(round(sum((ood < 0.5)*1)/len(ood),2))
        
    else:
        score = "-"
    return score
    

def per_class_preds(y_pred,numclass=4): #y_pred is a vector with scores for the entire entry
    return np.array([np.sum(y_pred==i) for i in range(numclass)])

def average_per_class_score(yprob):
    #return np.std(yprob, axis=0)
    #return np.max(yprob, axis=0)-np.min(yprob, axis=0)
    #return np.product(yprob, axis=0)
    return np.mean(yprob, axis=0)


def get_class(y_pred, labels): 
    
    c=np.argmax(y_pred)
    return labels[c] , round(y_pred[c],3)


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

def pred2string(predictions, vindex):
    '''
    converts preidctions along the sequence to a string
    predictions: vector of predicted classes for all windows'''
    string = ''
    tmp_N = 0
    tmp_V = 0
    index = vindex
    
    for j,i in  enumerate(predictions):
        if i == index:
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
