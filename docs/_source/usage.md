## Running Jaeger

##### CPU/GPU mode
---
Once the environment is properly set up, using Jaeger is straightforward. The program can accept both compressed and uncompressed .fasta files containing the contigs as input. It will output a table containing the predictions and various statistics calculated during runtime.

```
jaeger predict -i input_file.fasta -o output_dir --batch 128
```

To run jaeger with Singularity/Apptainer
```
singularity run --bind /path/to/wd --nv jaeger.sif jaeger predict -i path/to/wd/xxx.fna -o path/to/wd/out
```

##### multi-GPU mode
---

We provide a new program that allows users to automatically run multiple instances of Jaeger on several GPUs allowing maximum utilization of state-of-the-art hardware. This program accepts a file with a list of paths to all input FASTA files. **--ngpu** flag can be used to set the number of GPUs at your disposal. **--maxworkers** flag can be used to set the number of samples that should be processed parallaly per GPU. All other arguments remains similar to 'Jaeger' program.


```
# to generate a list of fasta files in a dir
ls ./files/*.fna | xargs realpath > input_file_list

# to process eight samples in parallel on two GPUs
jaeger_parallel -i input_file_list -o output_dir --batch 128 --maxworkers 4 --ngpu 2
```
```{admonition} Key Features:
:class: hint
- You can control the number of parallel computations using this parameter. By default it is set to 96. If you run into OOM errors, please consider setting the `--batch` option to a lower value. for example 96 is good enough for a graphics card with 4 Gb of memory.
```

#### What is in the output?
---

All predictions are summarized in a table located at ```output_dir/<input_file>_jaeger.tsv``` <br>

| contig_id                           | length | prediction | entropy | reliability_score | host_contam | prophage_contam | … | window_summary | terminal_repeats | repeat_length |
|------------------------------------|--------|------------|---------|-------------------|-------------|-----------------|---|----------------|------------------|---------------|
| NODE_1109_length_9622_cov_23.163…  | 9622   | phage      | 0.127   | 0.730             | False       | False           | … | 1V1n2V         | DTR              | 13            |
| NODE_1181_length_9275_cov_26.864…  | 9275   | phage      | 0.203   | 0.723             | False       | False           | … | 4V             | null             | null          |
| NODE_123_length_36569_cov_24.228…  | 36569  | phage      | 0.458   | 0.702             | False       | False           | … | 9V1n7V         | null             | null          |
| NODE_149_length_32942_cov_23.754…  | 32942  | phage      | 0.503   | 0.691             | False       | False           | … | 3V1n1n11V      | null             | null          |
| NODE_231_length_24276_cov_21.832…  | 24276  | phage      | 0.502   | 0.688             | False       | False           | … | 1V1n3V1n5V     | null             | null          |

<br>
This table provides information about various contigs in a metagenomic assembly. Each row represents a single contig, and the columns provide information about the contig's ID, length, prediction (phage or bacteria), entropy of the softmax distribution, reliability score, host/prophage contamination flags, per-class window counts and scores, and a summary of the windows. The `reliability_score` (0–1, higher is more confident) and `entropy` (0–2, lower is more confident) can be used to evaluate prediction confidence. The `window_summary` column shows the pattern of phage (V) and non-phage (n) windows along the contig. The `terminal_repeats` and `repeat_length` columns report detected terminal repeat structures (e.g., DTR = direct terminal repeats).
<br>


| Field | Explanation | Expected range |
|----------|----------|----------|
| contig_id  | header of the fasta record  | -  |
| length  | length of the sequence  | length >= fsize  |
| prediction  | final prediction for the sequence  | phage or bacteria  |
| entropy  | entropy of the softmax distribution | 0 (low uncertainity) - 2 (high uncertainity) |
| reliability_score | reliability score for the squence | 0 (high uncertainity) - 1 (low uncertainity) |
| host_contam | flag indicating potential host sequence contamination | True/False |
| prophage_contam | flag indicating potential prophage sequence contamination | True/False |
| #_xxx_windows  | number of windows predicted as xxx  | -  |
| #_xxx_score  | mean of logits of all windows  |   -  |
| #_xxx_var  | variance of logits of all windows  | -  |
| window_summary  | graphical summary of windows classified as phage and non-phage  |  -  |
| terminal_repeats | detected terminal repeat type (DTR, ITR, etc.) | - |
| repeat_length | length of the terminal repeat | - |


#### cmdline options
---


````
jaeger predict --help
````
````

Usage: jaeger predict [OPTIONS]

  Runs Jaeger on a dataset

Options:
  -i, --input PATH         Path to input file  [required]
  -o, --output TEXT        Path to output directory  [required]
  --fsize INTEGER          Length of the sliding window  [default: 2000]
  --stride INTEGER         The gap between two the sliding windows
                           (stride==fsize)  [default: 2000]
  -m, --model TEXT         Select a deep-learning model to use.  [default:
                           default]
  --model_path TEXT        Give the path to a model. overrides --model
  --config PATH            Path to Jaeger config file (e.g., when using
                           Apptainer or Docker)
  -p, --prophage           Extract and report prophage-like regions
  -s, --sensitivity FLOAT  Sensitivity of the prophage extraction algorithm
                           (0-4)  [default: 1.5]
  --lc INTEGER             Minimum contig length for prophage extraction
                           [default: 500000]
  --rc FLOAT               Minimum reliability score required to accept
                           predictions  [default: 0.1]
  --pc INTEGER             Minimum phage score required to accept predictions
                           [default: 3]
  --batch INTEGER          Parallel batch size, lower if GPU runs out of
                           memory  [default: 96]
  --workers INTEGER        Number of threads to use  [default: 4]
  --getalllogits           Writes window-wise scores to a .npy file
  --getsequences           Writes the putative phage sequences to a .fasta
                           file
  --cpu                    Ignore available GPUs and explicitly run on CPU
  --physicalid INTEGER     Set default GPU device ID for multi-GPU systems
                           [default: 0]
  --mem INTEGER            GPU memory limit  [default: 4]
  --getalllabels           Get predicted labels for Non-Viral contigs
  -v, --verbose            Verbosity level: -vv debug, -v info  [default: 1]
  -f, --overwrite          Overwrite existing files
  --help                   Show this message and exit.


````

```{note}
* The program expects the input file to be in .fasta format.
* The program uses a sliding window approach to scan the input sequences, so the stride argument determines how far the window will move after each scan.
* The batch argument determines how many sequences will be processed in parallel.
* The program is compatible with both CPU and GPU. By default, it will run on the GPU, but if the --cpu option is provided, it will use the specified number of threads for inference.
* The program uses a pre-trained neural network model for phage genome prediction.
* The --getalllabels option will output predicted labels for Non-Viral contigs, which can be useful for further analysis.
It's recommended to use the output of this program in conjunction with other methods for phage genome identification.
```



#### Python Library
---

> **Note:** The Python API (`jaeger.api`) is currently experimental and not available in the latest release. For programmatic access, we recommend using the CLI or calling Jaeger via `subprocess`.
>
> ```python
> import subprocess
> subprocess.run([
>     "jaeger", "predict",
>     "-i", "input.fasta",
>     "-o", "output_dir",
>     "--batch", "128"
> ])
> ```


#### Predicting prophages with Jaeger
---

```
jaeger predict -p -i NC_002695.fna -o outdir
```
The outdir will contain the following files
```
|____Escherichia_coli_O157-H7_prophages
| |____plots
| | |____NC_002695_Escherichia_coli_O157-H7_jaeger.pdf
| |____prophages_jaeger.tsv
|____Escherichia_coli_O157-H7_jaeger.log
|____Escherichia_coli_O157-H7_jaeger.tsv
```

users can find the following visulaization in the ```plots``` directory <br><br>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Yasas1994/Jaeger/assets/34155351/3efcd886-e45a-454f-9f61-53f954932b84"  width="500">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Yasas1994/Jaeger/assets/34155351/6acc1561-2c36-42c5-94ba-523721e902a5"  width="500">
  <img alt="light mode" src="https://github.com/Yasas1994/Jaeger/assets/34155351/6acc1561-2c36-42c5-94ba-523721e902a5">
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
