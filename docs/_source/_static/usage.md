---
#### Running Jaeger
---
##### CPU/GPU mode
Once the environment is properly set up, using Jaeger is straightforward. The program can accept both compressed and uncompressed .fasta files containing the contigs as input. It will output a table containing the predictions and various statistics calculated during runtime. 

```
jaeger run -i input_file.fasta -o output_dir --batch 128
```

To run jaeger with singularity
```
singularity run --bind /path/to/wd --nv jaeger_1.1.30.sif jaeger run -i path/to/wd/xxx.fna -o path/to/wd/out
```
##### multi-GPU mode

We provide a new program that allows users to automatically run multiple instances of Jaeger on several GPUs allowing maximum utilization of state-of-the-art hardware. This program accepts a file with a list of paths to all input FASTA files. **--ngpu** flag can be used to set the number of GPUs at your disposal. **--maxworkers** flag can be used to set the number of samples that should be processed parallaly per GPU. All other arguments remains similar to 'Jaeger' program.


```
# to generate a list of fasta files in a dir
ls ./files/*.fna | xargs realpath > input_file_list

# to process eight samples in parallel on two GPUs 
jaeger_parallel -i input_file_list -o output_dir --batch 128 --maxworkers 4 --ngpu 2
```

##### Selecting the batch parameter 

You can control the number of parallel computations using this parameter. By default it is set to 96. If you run into OOM errors, please consider setting the --bactch option to a lower value. for example 96 is good enough for a graphics card with 4 Gb of memory.


#### What is in the output?

All predictions are summarized in a table located at ```output_dir/<input_file>_default.jaeger.tsv```

```
┌───────────────────────────────────┬────────┬────────────┬─────────┬───┬─────────────┬────────────────┬──────────────────┬───────────────┐
│ contig_id                         ┆ length ┆ prediction ┆ entropy ┆ … ┆ Archaea_var ┆ window_summary ┆ terminal_repeats ┆ repeat_length │
╞═══════════════════════════════════╪════════╪════════════╪═════════╪═══╪═════════════╪════════════════╪══════════════════╪═══════════════╡
│ NODE_1109_length_9622_cov_23.163… ┆ 9622   ┆ Phage      ┆ 0.43    ┆ … ┆ 0.143       ┆ 1V1n2V         ┆ null             ┆ null          │
│ NODE_1181_length_9275_cov_26.864… ┆ 9275   ┆ Phage      ┆ 0.327   ┆ … ┆ 0.504       ┆ 4V             ┆ null             ┆ null          │
│ NODE_123_length_36569_cov_24.228… ┆ 36569  ┆ Phage      ┆ 0.503   ┆ … ┆ 1.554       ┆ 9V1n7V         ┆ null             ┆ null          │
│ NODE_149_length_32942_cov_23.754… ┆ 32942  ┆ Phage      ┆ 0.458   ┆ … ┆ 3.229       ┆ 3V1n1n11V      ┆ null             ┆ null          │
│ NODE_231_length_24276_cov_21.832… ┆ 24276  ┆ Phage      ┆ 0.502   ┆ … ┆ 1.467       ┆ 1V1n3V1n5V     ┆ null             ┆ null          │
└───────────────────────────────────┴────────┴────────────┴─────────┴───┴─────────────┴────────────────┴──────────────────┴───────────────┘

```

This table provides information about various contigs in a metagenomic assembly. Each row represents a single contig, and the columns provide information about the contig's ID, length, the number of windows identified as prokaryotic, viral, eukaryotic, and archaeal, the prediction of the contig (Phage or Non-phage), the score of the contig for each category (bacterial, viral, eukaryotic and archaeal), and a summary of the windows. The table can be used to identify potential phage sequences in the metagenomic assembly based on the prediction column. The score columns can be used to further evaluate the confidence of the prediction and the window summary column can be used to understand the count of windows that contributed to the final prediction.

##### List of all output fields
| Field | Explanation | Expected range |
|----------|----------|----------|
| contig_id  | header of the fasta record  | -  |
| length  | length of the sequence  | length >= fsize  |
| prediction  | final prediction for the sequence  | phage or non-phage  |
| entropy  | entropy of the softmax distribution | 0 (low uncertainity) - 2 (high uncertainity) |
| reliability_score | reliability score for the squence | 0 (high uncertainity) - 1 (low uncertainity) |
| #_xxx_windows  | number of windows predicted as xxx  | -  |
| #_xxx_score  | mean of logits of all windows  |   -  |
| #_xxx_var  | variance of logits of all windows  | -  |
| window_summary  | graphical summary of windows classified as phage and non-phage  |  -  |



#### Options


````
jaeger run --help
````
````

## Jaeger 1.1.30 (yet AnothEr phaGe idEntifier) Deep-learning based bacteriophage discovery 
https://github.com/Yasas1994/Jaeger.git
usage: jaeger run  -i INPUT -o OUTPUT

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input file
  -o OUTPUT, --output OUTPUT
                        path to output directory
  --fsize [FSIZE]       length of the sliding window (value must be 2^n). default:2048
  --stride [STRIDE]     stride of the sliding window. default:2048 (stride==fsize)
  -m {default,experimental_1,experimental_2}, --model {default,experimental_1,experimental_2}
                        select a deep-learning model to use. default:default
  -p, --prophage        extract and report prophage-like regions. default:False
  -s [SENSITIVITY], --sensitivity [SENSITIVITY]
                        sensitivity of the prophage extraction algorithm (between 0 - 4). default: 1.5
  --lc [LC]             minimum contig length to run prophage extraction algorithm. default: 500000 bp
  --rc [RC]             minium reliability score required to accept predictions. default: 0.2
  --pc [PC]             minium phage score required to accept predictions. default: 3
  --batch [BATCH]       parallel batch size, set to a lower value if your gpu runs out of memory. default:96
  --workers [WORKERS]   number of threads to use. default:4
  --getalllogits        writes window-wise scores to a .npy file
  --getsequences        writes the putative phage sequences to a .fasta file
  --cpu                 ignore available gpus and explicitly run jaeger on cpu. default: False
  --physicalid [PHYSICALID]
                        sets the default gpu device id (for multi-gpu systems). default: 0
  --getalllabels        get predicted labels for Non-Viral contigs. default: False
  -v, --verbose         Verbosity level : -vvv warning, -vv info, -v debug, (default info)

Misc. Options:
  -f, --overwrite       Overwrite existing files


````
---
#### Python Library
---
Jaeger can be integrated into python scripts using the jaegeraa python library as follows.
currently the predict function accepts 4 different input types.
1) Nucleotide sequence -> str
2) List of Nucleotide sequences -> list(str,str,..)
3) python file object -> (io.TextIOWrapper)
4) python generator object that yields Nucleotide sequences as str (types.GeneratorType)
5) Biopython Seq object

```python
from jaegeraa.api import Predictions

model=Predictor()
predictions=model.predict(input,stride=2048,fragsize=2048,batch=100)
model.predict()

```

returns a dictionary of lists in the following format

```python
{'contig_id': ['seq_0', 'seq_1'],
 'length': [19000, 10503],
 '#num_prok_windows': [0, 0],
 '#num_vir_windows': [9, 0],
 '#num_fun_windows': [0, 5],
 '#num_arch_windows': [0, 0],
 'prediction': ['Phage', 'Non-phage'],
 'bac_score': [-1.9552012549506292, -1.9441368103027343],
 'vir_score': [6.6312947273254395, -3.097817325592041],
 'fun_score': [-5.712721400790745, -0.6870137214660644],
 'arch_score': [-2.4369852013058133, -0.8941479325294495],
 'window_summary': ['9V', '5n']}
 
```
This dictionary can be easily converted to a pandas dataframe using DataFrame.from_dict() method
```python
import pandas as pd
df = DataFrame.from_dict(predictions)
```
---
#### Notes
---
* The program expects the input file to be in .fasta format.
* The program uses a sliding window approach to scan the input sequences, so the stride argument determines how far the window will move after each scan.
* The batch argument determines how many sequences will be processed in parallel.
* The program is compatible with both CPU and GPU. By default, it will run on the GPU, but if the --cpu option is provided, it will use the specified number of threads for inference.
* The program uses a pre-trained neural network model for phage genome prediction.
* The --getalllabels option will output predicted labels for Non-Viral contigs, which can be useful for further analysis.
It's recommended to use the output of this program in conjunction with other methods for phage genome identification.


---

#### Predicting prophages with Jaeger
---

```
jaeger run -p -i NC_002695.fna -o outdir 
```
The outdir will contain the following files
```
|____Escherichia_coli_O157-H7_prophages
| |____plots
| | |____NC_002695_Escherichia_coli_O157-H7_jaeger.pdf
| |____prophages_jaeger.tsv
|____Escherichia_coli_O157-H7_jaeger.log
|____Escherichia_coli_O157-H7_default_jaeger.tsv
```

users can find the following visulaization in the ```plots``` directory <br><br>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Yasas1994/Jaeger/assets/34155351/3efcd886-e45a-454f-9f61-53f954932b84"  width="500">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Yasas1994/Jaeger/assets/34155351/6acc1561-2c36-42c5-94ba-523721e902a5"  width="500">
  <img alt="dark mode" src="https://github.com/Yasas1994/Jaeger/assets/34155351/3efcd886-e45a-454f-9f61-53f954932b84">
</picture>
</p>

 <br><br>


list of prophage coordinates can be found in ```prophages_jaeger.tsv```
```
┌─────────────┬────────────┬──────────┬──────────┬───┬──────────┬────────┬────────────┬────────────┐
│ contig_id   ┆ alignment_ ┆ identiti ┆ identity ┆ … ┆ gc%      ┆ reject ┆ attL       ┆ attR       │
│             ┆ length     ┆ es       ┆          ┆   ┆          ┆        ┆            ┆            │
╞═════════════╪════════════╪══════════╪══════════╪═══╪══════════╪════════╪════════════╪════════════╡
│ NC_002695   ┆ 16.0       ┆ 16.0     ┆ 1.0      ┆ … ┆ 0.435049 ┆ false  ┆ GCACCATTTA ┆ GCACCATTTA │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ AATCAA     ┆ AATCAA     │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
│ NC_002695   ┆ 15.0       ┆ 15.0     ┆ 1.0      ┆ … ┆ 0.493497 ┆ false  ┆ GCTTTTTTAT ┆ GCTTTTTTAT │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ ACTAA      ┆ ACTAA      │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
│ NC_002695   ┆ 60.0       ┆ 60.0     ┆ 1.0      ┆ … ┆ 0.511819 ┆ false  ┆ TGGCGGAAGC ┆ TGGCGGAAGC │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ GCAGAGATTC ┆ GCAGAGATTC │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆ GAACTCTGGA ┆ GAACTCTGGA │
│             ┆            ┆          ┆          ┆   ┆          ┆        ┆ AC…        ┆ AC…        │
│ NC_002695   ┆ 16.0       ┆ 16.0     ┆ 1.0      ┆ … ┆ 0.499516 ┆ false  ┆ TTCTTTATTA ┆ TTCTTTATTA │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ CCGGCG     ┆ CCGGCG     │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
│ NC_002695   ┆ 14.0       ┆ 14.0     ┆ 1.0      ┆ … ┆ 0.529465 ┆ false  ┆ CGTCATCAAG ┆ CGTCATCAAG │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ TGCA       ┆ TGCA       │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
└─────────────┴────────────┴──────────┴──────────┴───┴──────────┴────────┴────────────┴────────────┘

```