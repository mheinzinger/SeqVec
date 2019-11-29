import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from seqvec.seqvec import get_embeddings, save_from_generator


def test_embedder():
    with TemporaryDirectory() as temp_dir:
        seq_dir = Path("test-data/sequences.fasta")
        path = Path(temp_dir) / "embeddings.npy"
        model_dir = Path("test-cache")
        embeddings_generator = get_embeddings(
            seq_dir, model_dir, "|", 1, False, False, 15000, True
        )
        save_from_generator(path, True, embeddings_generator)
        expected = np.load("test-data/embeddings.npy")
        actual = np.load(Path(temp_dir) / "embeddings.npy")
        assert np.allclose(expected, actual)

        expected = json.loads(Path("test-data/embeddings.json").read_text())
        actual = json.loads(Path(temp_dir).joinpath("embeddings.json").read_text())
        assert expected == actual
