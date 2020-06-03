from pathlib import Path
from typing import List, Tuple
from unittest import mock

import numpy

from seqvec.seqvec import get_embeddings

# lengths, cpu, success
LogType = List[Tuple[List[int], bool, bool]]


class MockElmoMemory:
    cpu: bool
    log: LogType

    def __init__(self, cpu: bool, log: LogType):
        self.cpu = cpu
        self.log = log

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        lengths = [len(sequence) for sequence in batch]
        if not self.cpu:
            if sum(len(i) for i in batch) > 18:
                self.log.append((lengths, False, False))
                raise RuntimeError(f"Too big for the GPU: {[len(i) for i in batch]}")
            else:
                self.log.append((lengths, False, True))
                return [numpy.zeros((3, length, 1024)) for length in lengths]
        else:
            assert (
                sum(len(i) for i in batch) > 18
            ), "This could have been done on the GPU"
            self.log.append((lengths, True, True))
            return [numpy.zeros((3, length, 1024)) for length in lengths]

    def embed_sentence(self, sentence: List[str]) -> numpy.ndarray:
        # That's what the original elmo does
        return self.embed_batch([sentence])[0]


def test_fallbacks(caplog):
    """ Check that the fallbacks to single sequence processing and/or the CPU are working.

    batchsize is 15, actual GPU limit 18, i.e. real values divided by 1000

    Procedure:
     * [7, 7, 7] Fails, passes with single sequence processing
     * [7, 8] Passes
     * [20] Fails, fails with single sequence processing, passes on the CPU
    """
    elmo_log: LogType = []
    with mock.patch(
        "seqvec.seqvec.get_elmo_model", lambda _, cpu: MockElmoMemory(cpu, elmo_log)
    ):
        fasta_file = Path("test-data/fallback_test_sequences.fasta")
        model_dir = Path()
        embeddings_generator = get_embeddings(
            fasta_file, model_dir, batchsize=15, layer="all", id_field=0
        )
        list(embeddings_generator)

    assert elmo_log == [
        ([7, 7, 7], False, False),
        ([7], False, True),
        ([7], False, True),
        ([7], False, True),
        ([7, 8], False, True),
        ([20], False, False),
        ([20], False, False),
        ([20], True, True),
    ]

    assert caplog.messages == [
        "Error processing batch of 3 sequences: Too big for the GPU: [7, 7, 7]",
        "Sequences in the failing batch: ['M7', 'I7', 'L7']",
        "Starting single sequence processing",
        "Error processing batch of 1 sequences: Too big for the GPU: [20]",
        "Sequences in the failing batch: ['T20']",
        "Starting single sequence processing",
        "RuntimeError for T20 with 20 residues: Too big for the GPU: [20]",
        "Single sequence processing failed. Switching to CPU now. This slows down the embedding process.",
    ]
