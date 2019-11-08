import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from seqvec_embedder.seqvec_embedder import get_embeddings


def test_embedder():
    with TemporaryDirectory() as temp_dir:
        get_embeddings(
            seq_dir=Path("test-data/sequences.fasta"),
            emb_path=Path(temp_dir) / "embeddings.npy",
            model_dir=Path("test-cache"),
            per_protein=True,
        )
        expected = np.load("test-data/embeddings.npy")
        actual = np.load(Path(temp_dir) / "embeddings.npy")
        assert np.allclose(expected, actual)

        expected = json.loads(Path("test-data/embeddings.json").read_text())
        actual = json.loads(Path(temp_dir).joinpath("embeddings.json").read_text())
        assert expected == actual
