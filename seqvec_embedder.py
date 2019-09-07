#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:00:37 2019

@author: mheinzinger
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def get_elmo_model( model_dir, cpu, verbose ):
    
    weights_path = model_dir / 'weights.hdf5'
    options_path = model_dir / 'options.json'

    # if no pre-trained model is available, yet --> download it
    if not (weights_path.exists() and options_path.exists()):
        if verbose: 
            print('No existing model found. Start downloading pre-trained SeqVec (~360MB)...')
        import urllib.request
        Path.mkdir(model_dir)
        repo_link    = 'http://rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec'
        options_link = repo_link +'/options.json'
        weights_link = repo_link +'/weights.hdf5'
        urllib.request.urlretrieve( options_link, options_path )
        urllib.request.urlretrieve( weights_link, weights_path )

    cuda_device = 0 if torch.cuda.is_available() and not cpu else -1
    return ElmoEmbedder( weight_file=weights_path, options_file=options_path, cuda_device=cuda_device )
        


def read_fasta( fasta_path, split_char, id_field ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper()
    return sequences


def process_embedding( embedding, per_protein ):
    '''
        Direct output of ELMo has shape (3,L,1024), with L being the protein's
        length, 3 being the number of layers used to train SeqVec (1 CharCNN, 2 LSTMs)
        and 1024 being a hyperparameter chosen to describe each amino acid.
        When a representation on residue level is required, you can sum
        over the first dimension, resulting in a tensor of size (L,1024).
        If you want to reduce each protein to a fixed-size vector, regardless of its
        length, you can average over dimension L.
    '''
    embedding = torch.tensor(embedding) # cast array to tensor
    embedding = embedding.sum(dim=0) # sum over residue-embeddings of all layers (3k->1k)
    if per_protein: # if embeddings are required on the level of whole proteins
        embedding = embedding.mean(dim=0)
    return embedding.cpu().detach().numpy() # cast to numpy array



def get_embeddings( seq_dir, emb_path, model_dir, split_char, id_field, cpu, 
                       max_chars, per_protein, verbose ):

    seq_dict = dict() 
    emb_dict = dict()
    
    ####################### Read in FASTA file ###############################
    if seq_dir.is_file(): # if single fasta file should be processed
        seq_dict = read_fasta( seq_dir, split_char, id_field )
    elif not seq_dict: # if a directory was provided: read all files 
        for seq_path in seq_dir.glob('**/*fasta*'):
            seq_dict = merge_two_dicts( seq_dict, read_fasta(seq_path, split_char, id_field))
    
    ####################### Sort sequences ###############################
    # Sorting sequences according to length is crucial for speed as batches 
    # of proteins with similar size increase throughput.
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ) )
    
    if verbose: print('Total number of sequences: {}'.format(len(seq_dict)))

    model = get_elmo_model( model_dir, cpu, verbose )
        
    batch          = list()
    length_counter = 0
    
    for index, (identifier, sequence) in enumerate(seq_dict): # for all sequences in the set

        # append sequence to batch and sum amino acids over proteins in batch
        batch.append( (identifier, sequence) )
        length_counter += len(sequence)
        
        # Transform list of batches to embeddings
        # if a) max. number of chars. for a batch is reached, 
        # if b) sequence is longer than half  max_chars (avoids runtimeError for very long seqs.)
        # if c) the last sequence is reached
        if length_counter > max_chars or len(sequence)>max_chars/2 or index==len(seq_dict)-1:

            # create List[List[str]] for batch-processing of ELMo
            tokens = [ list(seq) for _, seq in batch]
            embeddings = model.embed_sentences(tokens)
            
            #######################  Batch-Processing ####################### 
            runtime_error = False
            for batch_idx, (sample_id, seq) in enumerate(batch): # for each seq in the batch
                try: # try to get the embedding for the current sequnce
                    embedding = next(embeddings)
                except RuntimeError:
                    if verbose:
                        print('RuntimeError for {} (len={}).'.format(sample_id,len(seq)))
                        print('Starting single sequence processing')
                    runtime_error = True
                    break
                
                # if protein was embedded successfully --> save embedding
                embedding = process_embedding( embedding, per_protein )
                emb_dict[sample_id] = embedding

            ################## Single Sequence Processing ####################
            # Single sequence processing in case of runtime error due to 
            # a) very long sequence or b) too large batch size
            # If this fails, you might want to consider lowering max_chars and/or 
            # cutting very long sequences into smaller chunks
            if runtime_error:
                for batch_idx, (sample_id, seq) in enumerate(batch):
                    try:
                        embedding = model.embed_sentence( tokens[batch_idx] )
                    except RuntimeError:
                        print('RuntimeError for {} (len={}).'.format(sample_id,len(seq)))
                        print('Single sequence processing not possible. Skipping seq. ..' + 
                              'Consider splitting the sequence into smaller seqs or process on CPU.')
                        continue
                    
                    # if protein was embedded successfully --> save embedding
                    embedding = process_embedding( embedding, per_protein )
                    emb_dict[sample_id] = embedding
            
            ################## Reset batch ####################
            batch = list()
            length_counter = 0
            if verbose: print('.', flush=True, end='')

    if verbose: print('\nTotal number of embeddings: {}'.format(len(emb_dict)))

    ################## Write embeddings to file ####################
    try:
        if verbose: print('Writing embeddings to: {}'.format(emb_path))
        # save elmo representations    
        np.savez( emb_path, **emb_dict)
    except ZeroDivisionError:
        print('Error: Embedding dictionary is empty!')

    return None


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'embedder.py creates ELMo embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )
    
    # Path to fasta file (required)
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).' + 
                            'Can also be a directory holding multiple fasta files.')

    # Path for writing embeddings (required)
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path to a file for saving the created embeddings as NumPy .npz file.')

    # Path to model (optoinal)
    parser.add_argument('--model', type=str, 
                    default=Path.cwd() / 'model',
                    help='A path to a directory holding a pre-trained ELMo model. '+
                        'If the model is not found in this path, it will be downloaded automatically.' +
                        'The file containing the weights of the model must be named weights.hdf5.' + 
                        'The file containing the options of the model must be named options.json')
    
    # Create embeddings for a single protein or for all residues within a protein
    parser.add_argument('--protein', type=bool, 
                    default=False,
                    help='Flag for summarizing embeddings from residue level to protein level ' +
                    'via averaging. Default: False')
    
    # Number of residues within one batch
    parser.add_argument('--batchsize', type=int, 
                    default=15000,
                    help='Number of residues which need to be accumulated before starting batch ' + 
                    'processing. If you encounter an OutOfMemoryError, lower this value. Default: 15000')
    
    # Character for splitting fasta header
    parser.add_argument('--split_char', type=str, 
                    default='|',
                    help='The character for splitting the FASTA header in order to retrieve ' +
                        "the protein identifier. Should be used in conjunction with --id." +
                        "Default: '|' ")
    
    # Field index for protein identifier in fasta header after splitting with --split_char 
    parser.add_argument('--id', type=int, 
                    default=0,
                    help='The index for the uniprot identifier field after splitting the ' +
                        "FASTA header after each symbole in ['|', '#', ':', ' ']." +
                        'Default: 1')
    
    # Whether to use CPU or GPU
    parser.add_argument('--cpu', type=bool, 
                    default=False,
                    help='Flag for using CPU to compute embeddings. Default: False')
    
    # Whether to print some statistics while processing
    parser.add_argument('--verbose', type=bool, 
                    default=True,
                    help='Embedder gives some information while processing. Default: True')
    return parser


def main():
    
    parser = create_arg_parser()

    args = parser.parse_args()
    seq_dir   = Path( args.input )
    emb_path  = Path( args.output)
    model_dir = Path( args.model )
    split_char= args.split_char
    id_field  = args.id
    cpu_flag  = args.cpu
    per_prot  = args.protein
    max_chars = args.batchsize
    verbose   = args.verbose
    
    get_embeddings( seq_dir, emb_path, model_dir, split_char, id_field, 
                       cpu_flag, max_chars, per_prot, verbose )


if __name__ == '__main__':
    main()
