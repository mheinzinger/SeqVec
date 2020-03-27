import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from seqvec.seqvec import get_embeddings, save_from_generator

SPLIT_CHAR = "|"
ID_FIELD = 1
CPU_FLAG = False
SUM_LAYERS = True
MAX_CHARS = 15000
PER_PROTEIN = True

def test_embedder():
    with TemporaryDirectory() as temp_dir:
        seq_dir = Path("test-data/sequences.fasta")
        path = Path(temp_dir) / "embeddings.npy"
        model_dir = Path("test-cache")
        embeddings_generator = get_embeddings(
            seq_dir, model_dir, SPLIT_CHAR, ID_FIELD, CPU_FLAG, SUM_LAYERS, MAX_CHARS, PER_PROTEIN
        )
        save_from_generator(path, PER_PROTEIN, embeddings_generator)
        expected = np.load("test-data/embeddings.npy")
        actual = np.load(Path(temp_dir) / "embeddings.npy")
        assert np.allclose(expected, actual)

        expected = json.loads(Path("test-data/embeddings.json").read_text())
        actual = json.loads(Path(temp_dir).joinpath("embeddings.json").read_text())
        assert expected == actual
