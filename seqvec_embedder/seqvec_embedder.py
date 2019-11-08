#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_elmo_model(model_dir: Path, cpu: bool) -> ElmoEmbedder:
    weights_path = model_dir / "weights.hdf5"
    options_path = model_dir / "options.json"

    # if no pre-trained model is available, yet --> download it
    if not (weights_path.exists() and options_path.exists()):
        logger.info(
            "No existing model found. Start downloading pre-trained SeqVec (~360MB)..."
        )

        Path.mkdir(model_dir, exist_ok=True)
        repo_link = "http://rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec"
        options_link = repo_link + "/options.json"
        weights_link = repo_link + "/weights.hdf5"
        urllib.request.urlretrieve(options_link, options_path)
        urllib.request.urlretrieve(weights_link, weights_path)

    cuda_device = 0 if torch.cuda.is_available() and not cpu else -1
    logger.info("Loading the model")
    # TODO: Remove the string casting once the typing fixes in allennlp are released
    # https://github.com/allenai/allennlp/pull/3358
    return ElmoEmbedder(
        weight_file=str(weights_path),
        options_file=str(options_path),
        cuda_device=cuda_device,
    )


def read_fasta(
    sequences: Dict[str, str], fasta_path: Path, split_char: str, id_field: int
):
    """ Reads in fasta file containing multiple sequences.
    Adds all sequencces to the `sequences` dictionary.
    """

    if fasta_path.suffix == ".gz":
        handle = gzip.open(str(fasta_path), "rt")
    else:
        handle = fasta_path.open()

    with handle as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith(">"):
                if id_field == -1:
                    uniprot_id = line.replace(">", "").strip()
                else:
                    uniprot_id = (
                        line.replace(">", "").strip().split(split_char)[id_field]
                    )
                sequences[uniprot_id] = ""
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[uniprot_id] += "".join(line.split()).upper()


def read_fasta_file(seq_dir: Path, split_char: str, id_field: int) -> Dict[str, str]:
    seq_dict = dict()
    """ Read in FASTA file """
    if seq_dir.is_file():  # if single fasta file should be processed
        read_fasta(seq_dict, seq_dir, split_char, id_field)
    else:  # if a directory was provided: read all files
        assert seq_dir.is_dir()
        for seq_path in seq_dir.glob("**/*fasta*"):
            read_fasta(seq_dict, seq_path, split_char, id_field)
    return seq_dict


def process_embedding(embedding: np.ndarray, per_protein: bool) -> np.ndarray:
    """
    Direct output of ELMo has shape (3,L,1024), with L being the protein's
    length, 3 being the number of layers used to train SeqVec (1 CharCNN, 2 LSTMs)
    and 1024 being a hyperparameter chosen to describe each amino acid.
    When a representation on residue level is required, you can sum
    over the first dimension, resulting in a tensor of size (L,1024).
    If you want to reduce each protein to a fixed-size vector, regardless of its
    length, you can average over dimension L.
    """
    embedding = torch.tensor(embedding)  # cast array to tensor
    # sum over residue-embeddings of all layers (3k->1k)
    embedding = embedding.sum(dim=0)
    if per_protein:  # if embeddings are required on the level of whole proteins
        embedding = embedding.mean(dim=0)
    return embedding.cpu().detach().numpy()  # cast to numpy array


def single_sequence_processing(
    batch: List[Tuple[str, str]], model: ElmoEmbedder
) -> Dict[str, np.ndarray]:
    """
    Single sequence processing in case of runtime error due to
    a) very long sequence or b) too large batch size
    If this fails, you might want to consider lowering max_chars and/or
    cutting very long sequences into smaller chunks

    Returns unprocessed embeddings
    """
    batch_results_dict = dict()
    for sample_id, seq in batch:
        try:
            embedding = model.embed_sentence(list(seq))
        except RuntimeError as e:
            logger.error(
                "RuntimeError for {} (len={}): {}".format(sample_id, len(seq), e)
            )
            logger.error(
                "Single sequence processing not possible. Skipping seq. .."
                + "Consider splitting the sequence into smaller seqs or process on CPU."
            )
            continue

        # if protein was embedded successfully --> save embedding
        batch_results_dict[sample_id] = embedding
    return batch_results_dict


def get_embeddings(
    seq_dir: Path,
    emb_path: Path,
    model_dir: Path,
    split_char: str = "|",
    id_field: int = 1,
    cpu: bool = False,
    max_chars: int = 15000,
    per_protein: bool = False,
):
    emb_dict = dict()

    seq_dict = read_fasta_file(seq_dir, split_char, id_field)

    # Sort sequences
    # Sorting sequences according to length is crucial for speed as batches
    # of proteins with similar size increase throughput.
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]))

    logger.info("Total number of sequences: {}".format(len(seq_dict)))

    model = get_elmo_model(model_dir, cpu)

    batch = list()
    length_counter = 0

    for index, (identifier, sequence) in enumerate(
        tqdm(seq_dict)
    ):  # for all sequences in the set

        # append sequence to batch and sum amino acids over proteins in batch
        batch.append((identifier, sequence))
        length_counter += len(sequence)

        # Transform list of batches to embeddings
        # if a) max. number of chars. for a batch is reached,
        # if b) sequence is longer than half max_chars (avoids RuntimeError for very long seqs.)
        # if c) the last sequence is reached
        if not (
            length_counter > max_chars
            or len(sequence) > max_chars / 2
            or index == len(seq_dict) - 1
        ):
            continue

        # create List[List[str]] for batch-processing of ELMo
        tokens = [list(seq) for _, seq in batch]
        batch_ids = [identifier for identifier, _ in batch]
        try:  # try to get the embedding for the current sequnce
            embeddings = model.embed_batch(tokens)
            assert len(batch) == len(embeddings)
            batch_results = dict(zip(batch_ids, embeddings))
        except RuntimeError as e:
            logger.error(
                "Error processing batch of {} sequences: {}".format(len(batch), e)
            )
            logger.error("Sequences in the failing batch: {}".format(batch_ids))
            logger.error("Starting single sequence processing")
            batch_results = single_sequence_processing(batch, model)

        for sequence_id, embedding in batch_results.items():
            emb_dict[sequence_id] = process_embedding(embedding, per_protein)

        # Reset batch
        batch = list()
        length_counter = 0

    if not emb_dict:
        raise RuntimeError("Embedding dictionary is empty!")

    logger.info("Total number of embeddings: {}".format(len(emb_dict)))

    # Write embeddings to file
    logger.info(
        "Writing embeddings to {} and the ids to {}".format(
            emb_path, emb_path.with_suffix(".json")
        )
    )
    # save elmo representations
    with emb_path.with_suffix(".json").open("w") as id_file:
        json.dump(list(emb_dict.keys()), id_file)
    np.save(str(emb_path), np.asarray(list(emb_dict.values())))


def create_arg_parser():
    """ Creates and returns the ArgumentParser object. """

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description=(
            "seqvec_embedder.py creates ELMo embeddings for a given text "
            + " file containing sequence(s) in FASTA-format."
        )
    )

    # Path to fasta file (required)
    # noinspection PyTypeChecker
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="A path to a fasta-formatted text file containing protein sequence(s)."
        + "Can also be a directory holding multiple fasta files.",
    )

    # Path for writing embeddings (required)
    # noinspection PyTypeChecker
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="A path to a file for saving the created embeddings as NumPy .npy file. "
        + "A .json file with the sequence ids will be created next to this file",
    )

    # Path to model (optoinal)
    # noinspection PyTypeChecker
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.cwd() / "model",
        help="A path to a directory holding a pre-trained ELMo model. "
        + "If the model is not found in this path, it will be downloaded automatically."
        + "The file containing the weights of the model must be named weights.hdf5."
        + "The file containing the options of the model must be named options.json",
    )

    # Create embeddings for a single protein or for all residues within a protein
    parser.add_argument(
        "--protein",
        type=bool,
        default=False,
        help="Flag for summarizing embeddings from residue level to protein level "
        + "via averaging. Default: False",
    )

    # Number of residues within one batch
    parser.add_argument(
        "--batchsize",
        type=int,
        default=15000,
        help="Number of residues which need to be accumulated before starting batch "
        + "processing. If you encounter an OutOfMemoryError, lower this value. Default: 15000",
    )

    # Character for splitting fasta header
    parser.add_argument(
        "--split-char",
        type=str,
        default="|",
        help="The character for splitting the FASTA header in order to retrieve "
        + "the protein identifier. Should be used in conjunction with --id. "
        + "Default: '|' ",
    )

    # Field index for protein identifier in fasta header after splitting with --split-char
    parser.add_argument(
        "--id",
        type=int,
        default=1,
        help="The index for the uniprot identifier field after splitting the "
        + "FASTA header after each symbole in ['|', '#', ':', ' ']. "
        + "Use -1 to deactivate splitting"
        + "Default: 1",
    )

    # Whether to use CPU or GPU
    parser.add_argument(
        "--cpu",
        type=bool,
        default=False,
        help="Flag for using CPU to compute embeddings. Default: False",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Embedder gives some information while processing. Default: True",
    )
    return parser


def main():
    parser = create_arg_parser()

    args = parser.parse_args()
    seq_dir = args.input
    emb_path = args.output
    model_dir = args.model
    split_char = args.split_char
    id_field = args.id
    cpu_flag = args.cpu
    per_prot = args.protein
    max_chars = args.batchsize
    verbose = args.verbose

    if verbose:
        # Oterwise the default level is warning
        logger.setLevel(logging.INFO)

    get_embeddings(
        seq_dir,
        emb_path,
        model_dir,
        split_char,
        id_field,
        cpu_flag,
        max_chars,
        per_prot,
    )


if __name__ == "__main__":
    main()
