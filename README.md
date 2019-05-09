# SeqVec

Repository for the paper "Modelling the Language of Life - Deep Learning Protein Sequences". 
Holds pre-trained SeqVec model for creating embeddings for amino acid sequences.

# Requirements

*  Python>=3.6.1
*  torch>=0.4.1
*  allennlp

# Model availability
The ELMo model trained on UniRef50 (=SeqVec) is available at:
[SeqVec](https://rostlab.org/~deepppi/seqvec.zip)

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
