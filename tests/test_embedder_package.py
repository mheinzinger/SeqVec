import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest import mock

import numpy
import numpy as np
from numpy import ndarray

from seqvec.seqvec import get_embeddings
from seqvec.seqvec import save_from_generator

SPLIT_CHAR = "|"
ID_FIELD = 1
CPU_FLAG = False
SUM_LAYERS = True
BATCHSIZE = 15000
PER_PROTEIN = True


class MockElmoEmbedder:
    embedded_batch: List[ndarray]
    embedded_sentence: ndarray

    def __init__(self):
        self.embedded_batch = [
            numpy.load("test-data/embedded_batch0.npy"),
            numpy.load("test-data/embedded_batch1.npy"),
        ]
        self.embedded_sentence = numpy.load("test-data/embedded_sentence.npy")

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        if len(batch) == 2:  # First call
            return self.embedded_batch
        else:  # Second call
            raise RuntimeError("Go into single-sequence processing due to OOM")

    def embed_sentence(self, _sentence: List[str]) -> numpy.ndarray:
        return self.embedded_sentence


def get_elmo_model_mock(_model_dir: Path, _cpu: bool) -> MockElmoEmbedder:
    return MockElmoEmbedder()


def test_embedder():
    with TemporaryDirectory() as temp_dir:
        seq_dir = Path("test-data/sequences.fasta")
        path = Path(temp_dir) / "embeddings.npy"
        model_dir = Path("test-cache")
        embeddings_generator = get_embeddings(
            seq_dir,
            model_dir,
            SPLIT_CHAR,
            ID_FIELD,
            CPU_FLAG,
            SUM_LAYERS,
            BATCHSIZE,
            PER_PROTEIN,
        )
        save_from_generator(path, PER_PROTEIN, embeddings_generator)
        expected = np.load("test-data/embeddings.npy")
        actual = np.load(Path(temp_dir) / "embeddings.npy")
        assert np.allclose(expected, actual)

        expected = json.loads(Path("test-data/embeddings.json").read_text())
        actual = json.loads(Path(temp_dir).joinpath("embeddings.json").read_text())
        assert expected == actual


def test_single_sequence_processing():
    """ Check that the single sequence processing also performs the resizing postprocessing """
    with mock.patch("seqvec.seqvec.get_elmo_model", get_elmo_model_mock):
        seq_dir = Path("test-data/single_sequence_processing.fasta")
        model_dir = Path("test-cache")
        embeddings_generator = get_embeddings(
            seq_dir, model_dir, cpu=True, batchsize=400
        )
        for key, value in list(embeddings_generator):
            assert value.shape[1:] == (3072,)
