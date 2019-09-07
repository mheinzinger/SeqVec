# SeqVec

Repository for the paper [Modelling the Language of Life - Deep Learning Protein Sequences](https://www.biorxiv.org/content/10.1101/614313v2). 
Holds pre-trained SeqVec model for creating embeddings for amino acid sequences. Also, contains checkpoint for fine-tuning.

# Abstract
Background: One common task in Computational Biology is the prediction of aspects of protein function and structure from their amino acid sequence. For 26 years, most state-of-the-art approaches toward this end have been marrying machine learning and evolutionary information. The retrieval of related proteins from ever growing sequence databases is becoming so time-consuming that the analysis of entire proteomes becomes challenging. On top, evolutionary information is less powerful for small families, e.g. for proteins from the Dark Proteome.
Results: We introduce a novel way to represent protein sequences as continuous vectors (embeddings) by using the deep bi-directional model ELMo taken from natural language processing (NLP). The model has effectively captured the biophysical properties of protein sequences from unlabeled big data (UniRef50). After training, this knowledge is transferred to single protein sequences by predicting relevant sequence features. We refer to these new embeddings as SeqVec (Sequence-to-Vector) and demonstrate their effectiveness by training simple convolutional neural networks on existing data sets for two completely different prediction tasks. At the per-residue level, we significantly improved secondary structure (for NetSurfP-2.0 data set: Q3=79%±1, Q8=68%±1) and disorder predictions (MCC=0.59±0.03) over methods not using evolutionary information. At the per-protein level, we predicted subcellular localization in ten classes (for DeepLoc data set: Q10=68%±1) and distinguished membrane- bound from water-soluble proteins (Q2= 87%±1). All results built upon the embeddings gained from the new tool SeqVec neither explicitly nor implicitly using evolutionary information. Nevertheless, it improved over some methods using such information. Where the lightning-fast HHblits needed on average about two minutes to generate the evolutionary information for a target protein, SeqVec created the vector representation on average in 0.03 seconds.
Conclusion: We have shown that transfer learning can be used to capture biochemical or biophysical properties of protein sequences from large unlabeled sequence databases. The effectiveness of the proposed approach was showcased for different prediction tasks using only single protein sequences. SeqVec embeddings enable predictions that outperform even some methods using evolutionary information. Thus, they prove to condense the underlying principles of protein sequences. This might be the first step towards competitive predictions based only on single protein sequences.

# Requirements

*  Python>=3.6.1
*  torch>=0.4.1
*  allennlp

# Model availability
The ELMo model trained on UniRef50 (=SeqVec) is available at:
[SeqVec-model](https://rostlab.org/~deepppi/seqvec.zip)

The checkpoint for the pre-trained model is available at:
[SeqVec-checkpoint](https://rostlab.org/~deepppi/seqvec_checkpoint.tar.gz)

# Example
For a general example on how to extract embeddings using ELMo, please check the 
official allennlp ELMo website: [ELMo-Tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)

Short example:


Load pre-trained model:

```
> from allennlp.commands.elmo import ElmoEmbedder
> from pathlib import Path
> model_dir = Path('path_to_pretrained_SeqVec_directory')
> weights = model_dir / 'weights.hdf5'
> options = model_dir / 'options.json'
> seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
```

Get embedding for amino acid sequence:

```
> seq = 'SEQWENCE' # your amino acid sequence
> embedding = seqvec.embed_sentence( list(seq) ) # List-of-Lists with shape [3,L,1024]
```

Batch embed sequences:

```
> seq1 = 'SEQWENCE' # your amino acid sequence
> seq2 = 'PROTEIN'
> seqs = [ list(seq1), list(seq2) ]
> seqs.sort(key=len) # sorting is crucial for speed
> embedding = seqvec.embed_sentences( seqs ) # returns: List-of-Lists with shape [3,L,1024]
```

Get 1024-dimensional embedding for per-residue predictions:

```
> import torch
> residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
```

Get 1024-dimensional embedding for per-protein predictions:
```
> protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]
```

# Availability of task-specific predictions using SeqVec-based models
[SeqVec predictions - Chris' Protein properties](https://embed.protein.properties/)
