# Funcom
Funcom Source Code Summarization Tool - Public Release

This repository contains the public release code for Funcom, a tool for source code summarization.  Code summarization is the task of automatically generating natural language descriptions of source code.

### Publications related to this work include:

LeClair, A., McMillan, C., "Recommendations for Datasets for Source Code Summarization", in Proc. of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL'19), Short Research Paper Track, Minneapolis, USA, June 2-7, 2019.

LeClair, A., Jiang, S., McMillan, C., "A Neural Model for Generating Natural Language Summaries of Program Subroutines", in Proc. of the 41st ACE/IEEE International Conference on Software Engineering (ICSE'19), Montreal, QC, Canada, May 25-31, 2019.  
https://arxiv.org/abs/1902.01954

### Example Output
Randomly sampled example output from the ast-attendgru model compared to reference good human-written summaries:

PROTOTYPE OUTPUT - HUMAN REFERENCE  
returns the duration of the movie - get the full length of this movie in seconds  
write a string to the client - write a string to all the connected clients  
this method is called to indicate the next page in the page - call to explicitly go to the next page from within a single draw  
returns a list of all the ids that match the given gene - get a list of superfamily ids for a gene name  
compares two nodes by their UNK - compare nodes n1 and n2 by their dx entry  
this method updates the tree panel - updates the tree panel with a new tree  
returns the number of residues in the sequence - get number of interacting residues in domain b  
returns true if the network is found - return true if passed inet address match a network which was used  
log status message - log the status of the current message as info  

## Update Notice

This repository is archival for the ICSE'19 paper mentioned above.  It is a good place to get started, but you may also want to look at our newer projects:

https://github.com/aakashba/callcon-public

https://github.com/Attn-to-FC/Attn-to-FC

https://github.com/acleclair/ICPC2020_GNN

## USAGE

### Step 0: Dependencies

We assume Ubuntu 18.04, Python 3.6, Keras 2.2.4, TensorFlow 1.12.  Your milage may vary on different systems.

### Step 1: Obtain Dataset

We provide a dataset of 2.1m Java methods and method comments, already cleaned and separated into training/val/test sets:  

https://s3.us-east-2.amazonaws.com/icse2018/index.html  

(Note: this paper is now several years old.  Please see an update of data here: https://github.com/aakashba/callcon-public)  

Extract the dataset to a directory (/scratch/ is the assumed default) so that you have a directory structure:  
/scratch/funcom/data/standard/dataset.pkl  
etc. in accordance with the files described on the site above.

To be consistent with defaults, create the following directories:  
/scratch/funcom/data/outdir/models/  
/scratch/funcom/data/outdir/histories/  
/scratch/funcom/data/outdir/predictions/  

### Step 2: Train a Model

```console
you@server:~/dev/funcom$ time python3 train.py --model-type=attendgru --gpu=0
```

Model types are defined in model.py.  The ICSE'19 version is ast-attendgru, if you are seeking to reproduce it for comparision to your own models.  Note that history information for each epoch is stored in a pkl file e.g. /scratch/funcom/data/outdir/histories/attendgru_hist_1551297717.pkl.  The integer at the end of the file is the Epoch time at which training started, and is used to connect history, configuration, model, and prediction data.  For example, training attendgru to epoch 5 would produce:

/scratch/funcom/data/outdir/histories/attendgru_conf_1551297717.pkl  
/scratch/funcom/data/outdir/histories/attendgru_hist_1551297717.pkl  
/scratch/funcom/data/outdir/models/attendgru_E01_1551297717.h5  
/scratch/funcom/data/outdir/models/attendgru_E02_1551297717.h5  
/scratch/funcom/data/outdir/models/attendgru_E03_1551297717.h5  
/scratch/funcom/data/outdir/models/attendgru_E04_1551297717.h5  
/scratch/funcom/data/outdir/models/attendgru_E05_1551297717.h5  

A good baseline for initial work is the attendgru model.  Comments in the file (models/attendgru.py) explain its behavior in detail, and it trains relatively quickly: about 45 minutes per epoch using batch size 200 on a single Quadro P5000, with maximum performance on the validation set at epoch 5.

### Step 3: Inference / Prediction

```console
you@server:~/dev/funcom$ time python3 predict.py /scratch/funcom/data/outdir/models/attendgru_E05_1551297717.h5 --gpu=0
```

The only necessary input to predict.py on the command line is the model file, but configuration information is read from the pkl files mentioned above.  Output predictions will be written to a file e.g.:

/scratch/funcom/data/outdir/predictions/predict-attendgru_E05_1551297717.txt

Note that CPU prediction is possible in principle, but by default the attendgru and ast-attendgru models use CuDNNGRU instead of standard GRU, which necessitates using a GPU during prediction.

### Step 4: Calculate Metrics

```console
you@server:~/dev/funcom$ time python3 bleu.py /scratch/funcom/data/outdir/predictions/predict-attendgru_E05_1551297717.txt
```

This will output a BLEU score for the prediction file.
