import os
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ruptures as rpt
import parasail
from kneed import KneeLocator
import scipy
# logging_module.py
import logging
import sys


logger = logging.getLogger(__name__)

#https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        #return run_values, run_starts, run_lengths
        return run_values, run_lengths

# code for generation the window summaries
# find_runs fn retrives the lengths of consecutive items (items, run_lengths)
#vectorized function to generate window summaries

def get_window_summary(x, phage_pos = 1):
    items, run_length = find_runs(x)
    run_length = np.array(run_length, dtype=np.unicode_)
    tmp = np.empty(items.shape,dtype=np.unicode_)
    tmp[items != 1] = 'n'
    tmp[items == 1] = 'V'
    x=np.char.add(run_length, tmp)
    return ''.join(x)

def update_dict(x, num_classes = 4):
    d = {i:0 for i in range(num_classes)}
    d.update(dict(zip(x[0],x[1])))
    return d
def shanon_entropy(p): #aka information gain
    p = np.array(p)
    result = np.where(p > 0.0000000001, p, -10)
    p_log = np.log2(result, out=result, where=result > 0)
    return -np.sum(p*p_log, axis=-1)

def softmax_entropy(x):
    ex = np.exp(x)
    return shanon_entropy(ex/np.sum(ex, axis=-1).reshape(-1,1))

def smoothen_scores(x,w=5):
    bac = np.convolve(x[:,0],np.ones(w)/w, mode='same')
    phg = np.convolve(x[:,1],np.ones(w)/w, mode='same')
    euk = np.convolve(x[:,2],np.ones(w)/w, mode='same')
    arch = np.convolve(x[:,3],np.ones(w)/w, mode='same')
    return np.squeeze(np.dstack([bac,phg,euk, arch]))



def ood_predict(x_features, params):
    #use parameters extimated using sklearn
    x_features = (x_features- x_features.mean(axis=-1).reshape(-1,1))/x_features.std(axis=-1).reshape(-1,1)
    logits = np.dot(x_features,params['coeff'].reshape(-1,1)) + params['intercept']
    return (1/(1+ np.exp(logits))).flatten(), logits

def normalize(x):
    x_mean = x.mean(axis=1).reshape(-1,1)
    x_std = x.std(axis=1).reshape(-1,1) 
    return (x- x_mean)/x_std

def normalize_with_batch_stats(x, mean, std):
    return  (x - mean)/std

def normalize_l2(x):
    return x/np.linalg.norm(x,2, axis=1).reshape(-1,1)

def ood_predict_default(x_features, params):
    #use parameters extimated using sklearn
    if params['type'] == 'params':
        x_features = normalize_with_batch_stats(x_features, params['batch_mean']),params['batch_std']
        x_features = normalize(x_features)
        logits = np.dot(x_features,params['coeff'].reshape(-1,1)) + params['intercept']
        return (1/(1+ np.exp(-logits))).flatten(), logits
    #use a saved a sklearn model
    elif params['type'] == 'sklearn':
        features_data = normalize_with_batch_stats(x_features, params['batch_mean'], params['batch_std'])
        features_data_l2 = normalize_l2(features_data)
        
        return params['model'].predict_proba(features_data_l2)[:,0],0

def get_ood_probability(ood):
    if ood is not None:
        #print(ood)
        score = str(round(sum((ood < 0.5)*1)/len(ood),2))
        
    else:
        score = "-"
    return score
    
        # ev = np.product(np.argsort(data['pred_sum'],axis=1)[:,2:4]==np.array([2,1]), axis=1)
        # av = np.product(np.argsort(data['pred_sum'],axis=1)[:,2:4]==np.array([3,1]), axis=1)*2
        # bv = np.product(np.argsort(data['pred_sum'],axis=1)[:,2:4]==np.array([0,1]), axis=1)*3

        # class_map = config['default_labels'] #comes from config
        # class_map2 = config['second'] #comes from config

        # output = pd.DataFrame({
        #     'contig_id': headers,
        #     'length' : lengths,
        #     'prediction' : list(map(lambda x: class_map[x],consensus)),
        #     'prediction_2' : list(map(lambda x: class_map2[x],ev+av+bv)), #only for deafult mode

        #     '#_bac_windows': list(map(lambda x,index=0 : x[index], per_class_counts)),
        #     '#_phage_windows': list(map(lambda x,index=1 : x[index], per_class_counts)),
        #     '#_euk_windows': list(map(lambda x,index=2 : x[index], per_class_counts)),
        #     '#_arch_windows': list(map(lambda x,index=3 : x[index], per_class_counts)),

        #     'bac_score' :list(map(lambda x, index=0:x[index] ,pred_sum)),
        #     'bac_std' :list(map(lambda x, index=0:x[index] ,pred_std)),
        #     'phage_score' : list(map(lambda x, index=1:x[index] ,pred_sum)),
        #     'phage_std' :list(map(lambda x, index=1:x[index] ,pred_std)),
        #     'euk_score' :list(map(lambda x, index=2:x[index] ,pred_sum)) ,
        #     'euk_std' :list(map(lambda x, index=2:x[index] ,pred_std)),
        #     'arch_score':list(map(lambda x, index=3:x[index] ,pred_sum)),
        #     'arch_std' :list(map(lambda x, index=3:x[index] ,pred_std)),

        #     'window_summary': list(map(lambda x: get_window_summary(x)  ,frag_pred)),
            
        #     }
        # )
        

def write_output(args, config, data, output_file_path):
    try:
        #values = []
        #keys = config[args.model]['header']
        class_map = config['labels'] #comes from config

        lab = {int(k):v for k,v in config[args.model]['all_labels'].items()}

        #consider adding other infomation such as GC content in the future
        columns ={'contig_id': data['headers'],
                  'length' : data['length'],
                  'prediction' : list(map(lambda x: class_map[x],data['consensus'])),
                  'entropy' : data['entropy'],
                  'realiability_score' : list(map(lambda x: np.mean(x) ,data['ood'])),
                  'host_contam' : data['host_contam'],
                  'prophage_contam' : data['prophage_contam']
                  }

        if args.model == "deafult":
            # finds and appends the second highest class to the dict-> prediction_2
            ev = np.product(np.argsort(data['pred_sum'],axis=1)[:,2:4]==np.array([2,1]), axis=1)
            av = np.product(np.argsort(data['pred_sum'],axis=1)[:,2:4]==np.array([3,1]), axis=1)*2
            bv = np.product(np.argsort(data['pred_sum'],axis=1)[:,2:4]==np.array([0,1]), axis=1)*3
            class_map2 = {int(k):v for k,v in config[args.model]['second'].items()} #comes from config
            columns['prediction_2'] = list(map(lambda x: class_map2[x],ev+av+bv)) #only for deafult mode
            

        for i,label in lab.items():
            # appends the number of class-wise windows to the dict 
            columns[f'#_{label}_windows'] = list(map(lambda x,index=i : x[index], data['per_class_counts']))

        for i,label in lab.items():
            # appends the class-wise scores and score variance to the dict
            columns[f'{label}_score'] = list(map(lambda x, index=i : x[index] ,data['pred_sum']))
            columns[f'{label}_var'] =list(map(lambda x, index=i :x [index] ,data['pred_var']))
        # append the window_summary col to the dict
        columns['window_summary'] = list(map(lambda x, phage_pos=config[args.model]['vindex']: get_window_summary((x == 1)*1, phage_pos)  ,data['frag_pred']))
        

        df=pd.DataFrame(columns)
        df['host_contamination'] = df.apply(lambda x :(x['Phage_score'] < x['Phage_var'])*(x['prediction'] == 'Phage'), axis=1)
        df.to_csv(output_file_path, sep='\t', index=None, float_format='%.3f') 

    except Exception as e:
        logger.exception(e)
        sys.exit(1)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def merge_overlapping_ranges(intervals):
    """
    Merge overlapping ranges in a list of intervals.

    Args:
    - intervals: List of intervals, each represented as [start, end].

    Returns:
    - merged_intervals: List of merged intervals.
    """
    if len(intervals) == 0:
        return []

    # Convert the intervals to NumPy array for easier manipulation
    intervals = np.array(intervals)

    # Sort the intervals based on the start value
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]

    merged_intervals = [sorted_intervals[0]]

    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged_intervals[-1]

        if current_start <= last_end:  # Overlapping intervals
            merged_intervals[-1][1] = max(last_end, current_end)
        else:  # Non-overlapping intervals
            merged_intervals.append([current_start, current_end])

    return merged_intervals

def check_middle_number(array): #check this function for edge cases
    tmp = []
    for i in np.arange(len(array)-1):
        tmp.append(i+np.argmax(array[i:i+2]))
    mask = np.zeros(len(array), dtype=np.bool_)
    mask[tmp] = 1

    return mask


def logits_to_df(logits, headers, lengths, cutoff_length = 500_000):

    """
    Convert logits to a dict of dataframe.

    Args:
    - logits: list of numpy arrys
    - headers: numpy array of sequence identifiers of contigs

    Returns:
    - tmp : Dict of [pandas dataframes, str:host, int:lengths]
    """

    tmp={}
    for key,value,length in zip(headers,logits,lengths):
        #print(value.shape, length)
        if length >= cutoff_length:
            try:
                value = np.exp(value)/np.sum(np.exp(value), axis=1).reshape(-1,1)
                #value = (value - np.mean(value, axis=1).reshape(-1,1))/ np.std(value, axis=1).reshape(-1,1)
                #value = np.exp(value)/np.sum(np.exp(value))
                #print(value)
                #value=(value - value.mean())/value.std() #normalize logits
                max_class = np.argmax(np.mean(value,axis=0)) #bac, phage, euk,arch
                host = ''
                if max_class == 0:
                    print("Host: Bacteria")
                    host = 'Bacteria'
                    c_0= np.convolve(value[:,0],  np.ones(4), mode='same')
                    c_1= np.convolve(value[:,1],  np.ones(4), mode='same')
                    c_3 = np.convolve(value[:,3], np.ones(4), mode='same') 
                elif max_class == 3:
                    print("Host: Archaea")
                    host = 'Archaea'
                    c_0= np.convolve(value[:,3],  np.ones(4), mode='same')
                    c_1= np.convolve(value[:,1], np.ones(4), mode='same') 
                    c_3 = np.convolve(value[:,0], np.ones(4), mode='same')   
                else:
                    if max_class == 2:
                        host = 'Eukaryota'
                        print("Host: Eukayota")
                    c_0= np.zeros_like(value[:,1])        
                    c_1= np.zeros_like(value[:,1])  
                    c_3 = np.convolve(value[:,1], np.ones(4), mode='same')
                c_2 = np.convolve(value[:,2], np.ones(4), mode='same')  
                
                c_d = np.convolve(c_1 - (c_0) , np.ones(4), mode='same')  
                #c_d = np.convolve(c_d , np.ones(10)/10, mode='same')  
                t = pd.DataFrame(value,columns=['Bacterial','Phage','Eukaryotic','Archael'])
                t = t.assign(length=[i*2048 for i in range(len(t))])
                t['Absolute diff']=c_d
                t['c_0'] = c_0
                t['c_1'] = c_1
                t['c_2'] = c_2
                t['c_3'] = c_3
                tmp[f"{key}"] = [t,host,length]
            except Exception as e:
                print(e)
            
    return tmp

def plot_scores(logits_df, outdir):
    # quantile cut-off 0.975 (or 0.025 of the right tail)
    sns.set_style('whitegrid')
    #sns.set(font='Arial', font_scale=1.)
    for i in logits_df.keys():
        fig, axs = plt.subplots(4,sharex=True,figsize=(15,3.5))
        
        tmp,host,length=logits_df[i]
        fig.suptitle(f"{i.rsplit('_',1)[0]}")
        sns.lineplot(data=tmp[['c_0','length']],x='length',y='c_0',ax=axs[0],dashes=False, color=sns.color_palette()[2], label='bacteria', legend=False)
        sns.lineplot(data=tmp[['c_1','length']],x='length',y='c_1',ax=axs[1],dashes=False, color=sns.color_palette()[0], label='phage', legend=False)
        sns.lineplot(data=tmp[['c_2','length']],x='length',y='c_2',ax=axs[2],dashes=False, color=sns.color_palette()[3], label='eukaryota', legend=False)
        sns.lineplot(data=tmp[['c_3','length']],x='length',y='c_3',ax=axs[3],dashes=False, color=sns.color_palette()[1], label='archaea', legend=False)
        axs[0].set(ylim=(-0.15, 4.2))
        axs[1].set(ylim=(-0.15, 4.2))
        axs[2].set(ylim=(-0.15, 4.2))
        axs[3].set(ylim=(-0.15, 4.2))

        axs[1].scatter(tmp[tmp['c_1']> np.quantile(tmp['c_1'],q=0.975)]['length'],
                    tmp[tmp['c_1']> np.quantile(tmp['c_1'],q=0.975)]['c_1'],c="red")

        axs[3].set_xlabel('Genomic cordinate')
        
        plt.savefig(os.path.join(outdir, f'{i.rsplit("_",1)[0].replace(" ", "_")}.png'), bbox_inches='tight',dpi=150)
        plt.close()

def segment(logits_df, outdir, cutoff_length = 500_000, sensitivity=1.5):

    sns.set_style('whitegrid')
    #sns.set(font='Arial', font_scale=1.4)
    phage_cordinates = {}
    for key in logits_df.keys():
        # change point detection
        tmp,host,length=logits_df[key]
        if length > cutoff_length:
            bkpts = []
            for i in range(1,10):
                algo = rpt.KernelCPD(kernel="linear", min_size=3, jump=1).fit(
                    tmp['c_1'].to_numpy()
                ) 
                my_bkps = algo.predict(pen=i)
                if len(my_bkps) > 1:
                    bkpts.append(my_bkps)

                elif len(my_bkps) == 1 :
                    break
            if len(bkpts):
                bkpt_lens = np.array([len(b) for b in bkpts])
                kn = KneeLocator(bkpt_lens, [i for i in range(len(bkpts))], curve='convex', direction='decreasing')

            if kn.knee:
                bkpt_index =[len(b) for b in bkpts].index(kn.knee)
            else:
                bkpt_index = np.searchsorted(bkpt_lens, 1)
                if bkpt_index == len(bkpt_lens):
                    bkpt_index = None

            # if bkpt_index:  
            #     fig, axs = plt.subplots(1,sharex=True,figsize=(3.5,3.5))  
            #     sns.lineplot([len(b) for b in bkpts], ax=axs)
            #     axs.scatter([len(b) for b in bkpts].index(kn.knee), kn.knee, color="r", s=100)
            #     plt.savefig(os.path.join(outdir, f'{i.replace(" ", "_")}_knee.png'), bbox_inches='tight',dpi=150)
                
            all_high_indices = tmp[tmp['c_1']> np.quantile(tmp['c_1'],q=0.975)].index.to_numpy()
            ranges = [bkpts[bkpt_index][i:i+2] for i in range(len(bkpts[bkpt_index])-1)]
            range_scores = np.array([tmp.loc[s:e]['c_1'].mean() for s, e in ranges])
            range_mask = range_scores > sensitivity
            selected_range_scores= range_scores[range_mask]
            #print(range_scores, range_mask)
            
            selected_ranges = merge_overlapping_ranges(np.array(ranges)[range_mask])
            selected_ranges = np.array(selected_ranges)
            nw_bkpts = np.append(selected_ranges.flatten() ,tmp[['c_1']].to_numpy().shape[0])

            fig, ax_arr = rpt.display(tmp[['c_1']].to_numpy(), nw_bkpts, figsize=(10, 1.5))
            plt.savefig(os.path.join(outdir, f'{key.replace(" ", "_")}_segments.png'), bbox_inches='tight',dpi=150)
            plt.close()
            phage_cordinates[f"{key}"] = [selected_ranges,selected_range_scores]

    return phage_cordinates

def get_alignment_summary(result_object, seq_len,record, cordinates, phage_score,type_='DTR'):
    ltr_cutoff = 250
    
    if result_object.saturated:
        return 'saturated'
    else:
        result_object.end_query
        result_object.end_ref
        result_object.score
        
        alig_len = len(result_object.traceback.query)
        f_gaps = result_object.traceback.query.count('-')
        rc_gaps = result_object.traceback.ref.count('-')
        iden = result_object.traceback.comp.count('|')

        if type_ == 'ITR':
            #forward
            s_alig_end = cordinates['start'][0] + result_object.end_query + 1
            s_alig_start = s_alig_end - alig_len 
            #rev complement
            e_alig_start =  cordinates['end'][1]  - result_object.end_ref - 1
            e_alig_end = e_alig_start + alig_len 
        
        elif type_ == 'DTR':
            #forward 5'
            s_alig_end = cordinates['start'][0] + result_object.end_query
            s_alig_start = s_alig_end - alig_len + 1 
            #forward 3'
            e_alig_end =  cordinates['end'][0] + result_object.end_ref
            e_alig_start = e_alig_end - alig_len + 1

            if (s_alig_end - s_alig_start) >= ltr_cutoff:
                type_ = f'LTR_{type_}' 

        percentage_of_N = record[s_alig_start:e_alig_end].seq.count('N') / (e_alig_end - s_alig_start)
        gc_ = (record[s_alig_start:e_alig_end].seq.count('G')+ record[s_alig_start:e_alig_end].seq.count('C') ) / (e_alig_end - s_alig_start)

        return {'seq_id' : record.description,
                'alignment_length':alig_len, 
                'identities' : iden,
                'identity' :round(iden/alig_len,2),
                'score' : result_object.score,
                'type' : type_,
                'fgaps' : f_gaps,
                'rgaps' : rc_gaps,
                'sstart' : s_alig_start,
                'send' : s_alig_end,
                'estart' : e_alig_start,
                'eend' : e_alig_end, 
                'seq_len' : seq_len,
                'region_len':e_alig_end - s_alig_start,
                'phage_score':phage_score, 
                'n%': percentage_of_N,
                'gc%' : gc_,
                'reject' : True if percentage_of_N > 0.20 else False,
                'attL' : result_object.traceback.query,
                'attR' : result_object.traceback.ref,
                 }
def reverse_complement(dna_sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reverse_sequence = ''.join(complement_dict[base] for base in reversed(dna_sequence))
    return reverse_sequence

def get_cordinates(infile, phage_cordinates, outdir):
    #ideally DRs should be found in the intergenic region
    user_matrix = parasail.matrix_create("ACGT", 2, -100)
    summaries =[]
    for record in SeqIO.parse(infile, "fasta"):
        seq_len = len(record)
        
        if seq_len > 500_000:
            cord, scores = phage_cordinates.get(f"{record.description.split(',')[0]}", [[],[]])
            if len(cord) > 0  and len(scores) > 0:
                for i,j in zip(cord,scores):

                    start, end = i*2048
                
                    if (end - start)//2 < 14000:
                        off_set = ((end - start)//4)
                    else:
                        off_set = 2000
                    #print('searching for direct repeats',start-7000,start + off_set, '|', end-off_set,end + 7000, )

                    #TDR
                    result = parasail.sw_trace_scan_16( str(record.seq[start-7000:start + off_set]), 
                                                        str(record.seq[end - off_set: end + 7000 ]),
                                                        100, 5, user_matrix)

                    summaries.append(get_alignment_summary(result_object=result, 
                                                    seq_len=seq_len,
                                                    record = record,
                                                    cordinates = {'start':[start-7000,start + off_set],'end':[end - off_set , end + 7000 ]},
                                                    phage_score = j,
                                                    type_='DTR'))

    df = pd.DataFrame(summaries)
    df.to_csv(os.path.join(outdir,'prophages_jaeger.tsv'), sep='\t',index=False)