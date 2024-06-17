"""

Copyright (c) 2024 Yasas Wijesekara

"""

import os
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import ruptures as rpt
import parasail
from kneed import KneeLocator
import scipy
from pycirclize import Circos


# logging_module.py
import logging
import sys


logger = logging.getLogger(__name__)



def find_runs(x):
    #from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """Find runs of consecutive items in an array.
    
    Args
    ----
    x : list or np.array
        a list or a numpy array with integers

    Returns
    -------
    a tuple of lists (run_values, run_lengths)
    """
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

def get_window_summary(x, phage_pos):
    '''
    returns string representation of window-wise predictions

    Args
    ----

    x : list or np.array
        list or numpy array with window-wise integer class labels 

    phage_pos : int
        integer value representing Phage or Virus class

    Returns
    -------
    a string with Vs and ns. Vs represent virus or phage windows. ns represent cellular windows

    '''
    items, run_length = find_runs(x)
    run_length = np.array(run_length, dtype=np.unicode_)
    tmp = np.empty(items.shape,dtype=np.unicode_)
    #print(phage_pos, items, run_length)
    tmp[items != phage_pos] = 'n'
    tmp[items == phage_pos] = 'V'
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

def smoothen_scores(x, w=5):
    #smoothen the scores using moving average
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
        #x_features = normalize_with_batch_stats(x_features, params['batch_mean']),params['batch_std']
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

        class_map = config['labels'] #comes from config
        lab = {int(k):v for k,v in config[args.model]['all_labels'].items()}
        #consider adding other infomation such as GC content in the future
        columns ={'contig_id': data['headers'],
                  'length' : data['length'],
                  'prediction' : list(map(lambda x: class_map[x],data['consensus'])),
                  'entropy' : data['entropy'],
                  'reliability_score' : list(map(lambda x: np.mean(x) ,data['ood'])),
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
        columns['window_summary'] = list(map(lambda x, phage_pos=config[args.model]['vindex']: get_window_summary(x , phage_pos)  ,data['frag_pred']))
        
        df=pd.DataFrame(columns).set_index('contig_id')
        
        df = df.join(data['repeats'].set_index('contig_id')[['terminal_repeats','repeat_length']], how="outer").reset_index(names='contig_id')
        df['contig_id'] = df['contig_id'].apply(lambda x : x.replace('__', ','))
        df.to_csv(output_file_path, sep='\t', index=None, float_format='%.3f') 

    except Exception as e:
        logger.exception(e)

        sys.exit(1)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def merge_overlapping_ranges(intervals):
    """
    Merge overlapping ranges in a list of intervals.

    Args
    ----
        intervals: List of intervals, each represented as [start, end].

    Returns
    -------
        merged_intervals: List of merged intervals.
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
def scale_range (input, min, max):
    #min-max scaling
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

def gc_at_skew(seq : str, window:int=2048) -> dict:
    gc_skew = []
    #at_skew = []
    lengths = []
#deviation from [A] = [T] as (A − T)/(A + T)
#deviation from [C] = [G] as (C − G)/(C + G);
#lagging strand with negative GC skew.
    for i in range(0,len(seq)-window+1, window):
        g = seq.count("G", i, i+window)
        c = seq.count("C", i, i+window)
        #a = seq.count("A", i, i+window)
        #t = seq.count("T", i, i+window)

        #at_skew.append((a - t) / (a + t))
        gc_skew.append((g - c) / (g + c))
        lengths.append(i)
    gc_skew =  scale_range(np.convolve(np.array(gc_skew) , np.ones(10)/10, mode='same')  , min=-1, max=1)
    #at_skew =  scale_range(np.convolve(np.array(at_skew) , np.ones(10)/10, mode='same'), min=-1, max=1)
    cumsum = scale_range(np.cumsum(gc_skew), min=-1, max=1)
    return {'gc_skew': gc_skew, 'position':np.array(lengths), 'cum_gc': cumsum}



def logits_to_df(args, logits, headers, lengths, config, gc_skews, gcs, cutoff_length = 500_000):

    """
    Convert logits to a dict of dataframe.

    Args:
    - logits: list of numpy arrys
    - headers: numpy array of sequence identifiers of contigs

    Returns:
    - tmp : Dict of [pandas dataframes, str:host, int:lengths]
    """
    lab = {int(k):v for k,v in config[args.model]['all_labels'].items()}
    tmp={}
    for key,value,length, gc_skew, gc in zip(headers,logits,lengths, gc_skews, gcs):
        #print(value.shape, length)
        if length >= cutoff_length:
            try:
                value = np.exp(value)/np.sum(np.exp(value), axis=1).reshape(-1,1)
                #value = (value - np.mean(value, axis=1).reshape(-1,1))/ np.std(value, axis=1).reshape(-1,1)
                #value = np.exp(value)/np.sum(np.exp(value))
                #print(value)
                #value=(value - value.mean())/value.std() #normalize logits
                max_class = np.argmax(np.mean(value,axis=0)) #bac, phage, euk, arch
                host = lab[max_class]
                t = pd.DataFrame(value,columns=list(config[args.model]['all_labels'].values()))
                t = t.assign(length=[i*args.fsize for i in range(len(t))])

                for k,v in lab.items():
                    t[v] = np.convolve(value[:,k],  np.ones(4), mode='same')
                t['gc'] = gc
                t['gc_skew'] = scale_range(np.convolve(np.array(gc_skew) , np.ones(10)/10, mode='same')  , min=-1, max=1)
                # if max_class == 0:
                #     logger.info("Host: Bacteria")
                #     host = 'Bacteria'
                #     c_0 = np.convolve(value[:,0],  np.ones(4), mode='same')
                #     c_1 = np.convolve(value[:,1],  np.ones(4), mode='same')
                #     c_2 = np.convolve(value[:,2], np.ones(4), mode='same') 
                #     c_3 = np.convolve(value[:,3], np.ones(4), mode='same') 
                #     c_d = np.convolve(c_0 - (c_1) , np.ones(4), mode='same')
                # elif max_class == 3:
                #     logger.info("Host: Archaea")
                #     host = 'Archaea'
                #     c_0 = np.convolve(value[:,0],  np.ones(4), mode='same')
                #     c_1 = np.convolve(value[:,1], np.ones(4), mode='same') 
                #     c_2 = np.convolve(value[:,2], np.ones(4), mode='same') 
                #     c_3 = np.convolve(value[:,3], np.ones(4), mode='same')   
                #     c_d = np.convolve(c_3 - (c_1) , np.ones(4), mode='same')
                # elif max_class == 1:
                #     host = 'Phage'
                #     c_0 = np.convolve(value[:,0],  np.ones(4), mode='same')
                #     c_1 = np.convolve(value[:,1], np.ones(4), mode='same') 
                #     c_2 = np.convolve(value[:,2], np.ones(4), mode='same') 
                #     c_3 = np.convolve(value[:,3], np.ones(4), mode='same')   
                #     c_d = np.convolve(c_1 - (c_0) , np.ones(4), mode='same')
                # else:
                #     if max_class == 2:
                #         host = 'Eukaryota'
                #         logger.info("Host: Eukayota")
                #     c_0 = np.convolve(value[:,0],  np.ones(4), mode='same')
                #     c_1 = np.convolve(value[:,1], np.ones(4), mode='same') 
                #     c_2 = np.convolve(value[:,2], np.ones(4), mode='same') 
                #     c_3 = np.convolve(value[:,3], np.ones(4), mode='same')
                #     c_d = np.convolve(c_2 - (c_1) , np.ones(4), mode='same')

                   
                #c_d = np.convolve(c_d , np.ones(10)/10, mode='same')  

                # t['Absolute diff']=c_d
                # t['c_0'] = c_0
                # t['c_1'] = c_1
                # t['c_2'] = c_2
                # t['c_3'] = c_3
                tmp[f"{key}"] = [t,host,length]
            except Exception as e:
                logger.error(e)
            
    return tmp

def plot_scores(logits_df, args, config, outdir, phage_cordinates):
    '''
    Creates a circos plot of the host genome including the putative prophages 
    identified by Jaeger

    Args
    ----
    logits_df : pd.DataFrame

    phage_cordinates : dict

    args : dict

    config : dict

    outdir : str

    Returns
    -------
    
    '''
    # quantile cut-off 0.975 (or 0.025 of the right tail)
    # sns.set(font='Arial', font_scale=1.)
    lab = {int(k):v for k,v in config[args.model]['all_labels'].items()}
    legend_lines =[]

    #logits_df dict(contig_id -> (df, host, contig_length))
    for contig_id in logits_df.keys():

        tmp,host,length=logits_df[contig_id]
        circos = Circos(sectors={contig_id: length})
        #circos.text(f"{name[0].split('__')[0]}", size=12, r=20)
        sector = circos.get_sector(contig_id)

        # Plot outer track with xticks
        major_ticks_interval = 500_000
        minor_ticks_interval = 100_000
        outer_track = sector.add_track((98, 100))
        outer_track.axis(fc="lightgrey")
        outer_track.xticks_by_interval(
            major_ticks_interval, label_formatter=lambda v: f"{v/ 10 ** 6:.1f} Mb", show_endlabel=False, label_size=11
        )

        outer_track.xticks_by_interval(minor_ticks_interval, tick_length=1, show_label=False, label_size=11)
        colors = ['gray', 'green', 'red', 'teal', 'brown']
        patches = []

        for j,v in enumerate(lab.values()):
            # Plot Forward phage, bacterial, archaeal and eukaryotic scores
            if v == "Phage":
                phage_track = sector.add_track((88, 97), r_pad_ratio=0.1)
                phage_track.fill_between(tmp['length'],tmp[v].to_numpy(), vmin=0, vmax=4, color="orange", alpha=1)
                #phage_track.grid()

                for cords in phage_cordinates[contig_id][0]:
                    pcs = np.arange(cords[0], cords[-1])*args.fsize
                    phage_track.fill_between(pcs,np.ones_like(pcs)*4, vmin=0, vmax=4, color="magenta", alpha=0.3,lw=1)
            else:
                aux_track = sector.add_track((78, 87), r_pad_ratio=0.1)
                aux_track.fill_between(tmp['length'],  tmp[v].to_numpy(), vmin=0, vmax=4, color=colors[j], alpha=0.7)
                patches.append(Patch(color=colors[j], label=v))


            # sns.lineplot(data=tmp[[v,'length']],x='length',y=v,dashes=False, color=sns.color_palette()[j], legend=False)
            # if v == "Phage":
            #     axs[j].scatter(tmp[tmp[v]> np.quantile(tmp[v],q=0.975)]['length'],
            #         tmp[tmp[v]> np.quantile(tmp[v],q=0.975)][v],c="red")

        # Plot G+C
        gc_content_track = sector.add_track((55, 70))
        tmp['gc'] = tmp['gc'] - tmp['gc'].mean()
        positive_gc_contents = np.where(tmp['gc'] > 0, tmp['gc'], 0)
        negative_gc_contents = np.where(tmp['gc'] < 0, tmp['gc'], 0)
        abs_max_gc_content = np.max(np.abs(tmp['gc']))

        vmin, vmax = -abs_max_gc_content, abs_max_gc_content
        gc_content_track.fill_between(
            tmp['length'], positive_gc_contents, 0, vmin=vmin, vmax=vmax, color="blue", alpha=0.5
        )
        gc_content_track.fill_between(
            tmp['length'], negative_gc_contents, 0, vmin=vmin, vmax=vmax, color="black"
        )

        # Plot GC skew
        gc_skew_track = sector.add_track((45, 55))
        positive_gc_skews = np.where(tmp['gc_skew'] > 0, tmp['gc_skew'], 0)
        negative_gc_skews = np.where(tmp['gc_skew']< 0, tmp['gc_skew'], 0)
        abs_max_gc_skew = np.max(np.abs(tmp['gc_skew']))
        vmin, vmax = -abs_max_gc_skew, abs_max_gc_skew
        gc_skew_track.fill_between(
            tmp['length'], positive_gc_skews, 0, vmin=vmin, vmax=vmax, color="olive"
        )
        gc_skew_track.fill_between(
            tmp['length'], negative_gc_skews, 0, vmin=vmin, vmax=vmax, color="purple"
        )

        fig = circos.plotfig()
        plt.title(f"{contig_id.replace('__',',')}", fontdict={'size':14, 'weight':'bold'})
        # Add legend
        handles = [
            Patch(color="orange", label="Phage"),
            Patch(color="magenta",alpha=0.3, label="putative prophage"),
            ] + patches + [   
            Line2D([], [], color="blue", label="$ > \overline{G+C}$", marker="^", ms=6, ls="None", alpha=0.5),
            Line2D([], [], color="black", label="$ < \overline{G+C}$", marker="v", ms=6, ls="None"),
            Line2D([], [], color="olive", label="Positive GC Skew", marker="^", ms=6, ls="None"),
            Line2D([], [], color="purple", label="Negative GC Skew", marker="v", ms=6, ls="None"),
        ]
        _ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.51, 0.50), loc="center", fontsize=11)

        plt.savefig(os.path.join(outdir, f'{contig_id.rsplit("_",1)[0].replace(" ", "_")}.pdf'), bbox_inches='tight',dpi=300)
        plt.close()

def segment(logits_df, outdir, cutoff_length = 500_000, sensitivity=1.5):

    phage_cordinates = {}
    for key in logits_df.keys():
        # change point detection
        tmp,host,length=logits_df[key]
        if length > cutoff_length:
            try:
                bkpts = []
                algo = rpt.KernelCPD(kernel="linear", min_size=3, jump=1).fit(
                        tmp['Phage'].to_numpy()
                    ) 
                for i in range(1,10):

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
                    
                all_high_indices = tmp[tmp['Phage']> np.quantile(tmp['Phage'],q=0.975)].index.to_numpy()
                ranges = [bkpts[bkpt_index][i:i+2] for i in range(len(bkpts[bkpt_index])-1)]
                range_scores = np.array([tmp.loc[s:e]['Phage'].mean() for s, e in ranges])
                range_mask = range_scores > sensitivity
                selected_range_scores= range_scores[range_mask]
                #print(range_scores, range_mask)
                
                selected_ranges = merge_overlapping_ranges(np.array(ranges)[range_mask])
                selected_ranges = np.array(selected_ranges)
                nw_bkpts = np.append(selected_ranges.flatten() ,tmp[['Phage']].to_numpy().shape[0])

                #integrate prophage coordinates to the main figure
                #fig, ax_arr = rpt.display(tmp[['Phage']].to_numpy(), nw_bkpts, figsize=(10, 1.5))
                #plt.savefig(os.path.join(outdir, f'{key.replace(" ", "_")}_segments.png'), bbox_inches='tight',dpi=150)
                #plt.close()
                phage_cordinates[f"{key}"] = [selected_ranges,selected_range_scores]
            except Exception as e:
                phage_cordinates[f"{key}"] = [[],[]]
                logger.debug(e)

    return phage_cordinates

def get_prophage_summary(result_object, seq_len, record, cordinates, phage_score,type_='DTR'):
    ltr_cutoff = 250

    if result_object is None:

        percentage_of_N = None
        s_alig_start = cordinates['start'][0]
        e_alig_end = cordinates['end'][0]
        gc_ = (record[1][s_alig_start:e_alig_end].count('G')+ record[1][s_alig_start:e_alig_end].count('C')) / (e_alig_end - s_alig_start)

        return {'contig_id' : record[0],
                'alignment_length':None, 
                'identities' : None,
                'identity' :None,
                'score' : None,
                'type' : None,
                'fgaps' : None,
                'rgaps' : None,
                'sstart' : s_alig_start,
                'send' : None,
                'estart' : None,
                'eend' : e_alig_end, 
                'seq_len' : seq_len,
                'region_len':e_alig_end - s_alig_start,
                'phage_score':phage_score, 
                'n%': percentage_of_N,
                'gc%' : gc_,
                'reject' : None,
                'attL' : None,
                'attR' : None,
                 }
    elif result_object.saturated:
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

        percentage_of_N = record[1][s_alig_start:e_alig_end].count('N') / (e_alig_end - s_alig_start)
        gc_ = (record[1][s_alig_start:e_alig_end].count('G')+ record[1][s_alig_start:e_alig_end].count('C')) / (e_alig_end - s_alig_start)

        return {'contig_id' : record[0],
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
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', '-':'-', 'N':'N', 'W':'W','S':'S','Y':'R','R':'Y', 'M':'K',
                        'K':'M', 'B':'V', 'V':'B', 'H':'D', 'D':'H' }
    reverse_sequence = ''.join(complement_dict[base] for base in reversed(dna_sequence))
    return reverse_sequence

def get_cordinates(args, filehandle, prophage_cordinates, outdir):
    #should DR's be found in the intergenic regions?
    user_matrix = parasail.matrix_create("ACGT", 2, -100)
    summaries =[]
    filehandle.seek(0)
    for record in SeqIO.FastaIO.SimpleFastaParser(filehandle):
        seq_len = len(record[1])
        headder = record[0].replace(',','__')
        if seq_len > 500_000:
            cord, scores = prophage_cordinates.get(f"{headder}", [[],[]])
            if len(cord) > 0  and len(scores) > 0:
                for i,j in zip(cord,scores):

                    start, end = i*args.fsize
                    scan_length = min(max(int(seq_len*0.04),400), 4000)

                    if (end - start)//2 < 14000:
                        off_set = ((end - start)//4)
                    else:
                        off_set = 2000
                    logger.info('searching for direct repeats',start-scan_length,start + off_set, '|', end-off_set,end + scan_length, )

                    #TDR
                    result_dtr = parasail.sw_trace_scan_16( str(record[1][start-scan_length:start + off_set]), 
                                                        str(record[1][end - off_set: end + scan_length ]),
                                                        100, 5, user_matrix)
                    
                    result_itr = parasail.sw_trace_scan_16( str(record[1][start-scan_length:start + off_set]), 
                                                        reverse_complement(str(record[1][end - off_set: end + scan_length])),
                                                        100, 5, user_matrix)
                    
                    # summaries.append(get_prophage_summary(result_object=result, 
                    #                                       seq_len = seq_len,
                    #                                       record = record,
                    #                                       cordinates = {'start':[start-scan_length,start + off_set],
                    #                                                     'end':[end - off_set , end + scan_length ]},
                    #                                       phage_score = j,
                    #                                       type_='DTR'))
                    

                    # result_itr = parasail.sw_trace_scan_16( str(record[1][:scan_length]), 
                    #                                         reverse_complement(record[1][-scan_length:]),
                    #                                         100, 5, user_matrix)

                    # result_dtr = parasail.sw_trace_scan_16( str(record[1][:scan_length]), 
                    #                                         str(record[1][-scan_length:]),
                    #                                         100, 5, user_matrix)
                    
                    if len(result_itr.traceback.query) > 12 or len(result_dtr.traceback.query) > 12:

                        if result_itr.score > result_dtr.score:
                            summaries.append(get_prophage_summary(result_object=result_itr, 
                                                          seq_len = seq_len,
                                                          record = record,
                                                          cordinates = {'start':[start-scan_length,start + off_set],
                                                                        'end':[end - off_set , end + scan_length ]},
                                                          phage_score = j,
                                                          type_='ITR')) 
                        else:
                            summaries.append(get_prophage_summary(result_object=result_dtr, 
                                                          seq_len = seq_len,
                                                          record = record,
                                                          cordinates = {'start':[start-scan_length,start + off_set],
                                                                        'end':[end - off_set , end + scan_length ]},
                                                          phage_score = j,
                                                          type_='DTR')) 
                    else:
                        summaries.append(get_prophage_summary(result_object=None, 
                                                          seq_len = seq_len,
                                                          record = record,
                                                          cordinates = {'start':[start, None],
                                                                        'end':[end, None]},
                                                          phage_score = j,
                                                          type_=None))
    if len(summaries) > 0:
        df = pd.DataFrame(summaries)
        df['contig_id'] = df['contig_id'].apply(lambda x : x.replace('__', ','))
        df.to_csv(os.path.join(outdir,'prophages_jaeger.tsv'), sep='\t',index=False)

def safe_divide(numerator, denominator):
    try:
        result = round(numerator / denominator, 2)
    except ZeroDivisionError:
        logger.error("Error: Division by zero!")
        result = None
    return result

def get_alignment_summary(result_object, seq_len,record_id, input_length, type_='DTR'):
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
            s_alig_start = (result_object.end_query - alig_len + f_gaps) + 1
            s_alig_end = result_object.end_query + 1
            #rev complement
            e_alig_start = (seq_len - input_length) + max(input_length - result_object.end_ref ,0)
            e_alig_end = e_alig_start + (alig_len - rc_gaps)
            rear = reverse_complement(str(result_object.traceback.ref))
        
        elif type_ == 'DTR':
            #forward 5'
            s_alig_start = (result_object.end_query - alig_len + f_gaps) + 1
            s_alig_end = result_object.end_query + 1
            #forward 3'
            e_alig_start =(seq_len - input_length) + max(result_object.end_ref - alig_len, 0) 
            e_alig_end = (seq_len - input_length) + result_object.end_ref
            if (s_alig_end - s_alig_start) >= ltr_cutoff:
                type_ = f'LTR_{type_}' 
            rear = result_object.traceback.ref

        return {'contig_id' : record_id,
                'repeat_length':alig_len, 
                'identities' : iden,
                'identity' :safe_divide(iden,alig_len),
                'score' : result_object.score,
                'terminal_repeats' : type_,
                'fgaps' : f_gaps,
                'rgaps' : rc_gaps,
                'sstart' : s_alig_start,
                'send' : s_alig_end,
                'estart' : e_alig_start,
                'eend' : e_alig_end, 
                'seq_len' : seq_len,
                'front' : result_object.traceback.query,
                'rear' : rear,  }

def scan_for_terminal_repeats(infile):
    infile.seek(0)
    #ideally DRs should be found in the intergenic region
    logger.info("scaning for terminal repeats")
    user_matrix = parasail.matrix_create("ACGT", 2, -100)
    summaries =[]
    for record in SeqIO.FastaIO.SimpleFastaParser(infile):
        seq_len = len(record[1])
        headder = record[0].replace(',','__')
        scan_length = min(max(int(seq_len*0.04),400), 4000)

        result_itr = parasail.sw_trace_scan_16( str(record[1][:scan_length]), 
                                                reverse_complement(record[1][-scan_length:]),
                                                100, 5, user_matrix)

        result_dtr = parasail.sw_trace_scan_16( str(record[1][:scan_length]), 
                                                str(record[1][-scan_length:]),
                                                100, 5, user_matrix)
        if len(result_itr.traceback.query) > 12 or len(result_dtr.traceback.query) > 12:
            if result_itr.score > result_dtr.score:
                summaries.append(get_alignment_summary(result_object=result_itr, seq_len=seq_len, record_id = headder,input_length=scan_length, type_='ITR')) 
            else:
                summaries.append(get_alignment_summary(result_object=result_dtr, seq_len=seq_len, record_id = headder,input_length=scan_length, type_='DTR')) 
        else:
            summaries.append({'contig_id' : headder,
                'repeat_length':None, 
                'identities' : None,
                'identity' :None,
                'score' : None,
                'terminal_repeats' : None,
                'fgaps' : None,
                'rgaps' : None,
                'sstart' : None,
                'send' : None,
                'estart' : None,
                'eend' : None, 
                'seq_len' : seq_len,
                'front' : None,
                'rear' : None  }
)
    infile.seek(0)
    return pd.DataFrame(summaries)