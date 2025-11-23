"""Standalone HMM-only inference demo for Tiberius.

This script wires together the same building blocks used by the CLI
(`tiberius.main.run_tiberius`) but stops after Viterbi decoding and the
probability-based filtering step. It is meant for interactive debugging
in environments like Google Colab where you want to inspect the neural
network logits and HMM labels without creating a GTF/GFF file.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Dict, Iterable, List, Tuple

import importlib.metadata
import numpy as np
import tensorflow as tf

# Allow running the script directly from the repository root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _safe_version(name: str) -> str:
    """Return package version or a local placeholder if not installed."""

    try:
        return _ORIG_VERSION(name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0-local"


_ORIG_VERSION = importlib.metadata.version
importlib.metadata.version = _safe_version  # type: ignore[assignment]

from tiberius import PredictionGTF, lstm_model


@dataclasses.dataclass
class SimpleSeqRecord:
    """Minimal stand-in for BioPython's ``SeqRecord``.

    ``tiberius.genome_fasta.GenomeSequences`` expects objects with a ``seq``
    attribute. Using this tiny data class avoids an external BioPython
    dependency while matching the interface that ``PredictionGTF.init_fasta``
    forwards to ``GenomeSequences``.
    """

    seq: str


class HmmOnlyPipeline:
    """Replicate the DNA→CNN/BiLSTM→HMM path from ``PredictionGTF``.

    The methods inside mirror the calls that ``run_tiberius`` performs:

    * Build a CNN/BiLSTM backbone with ``tiberius.models.lstm_model``.
    * Attach a default ``GenePredHMMLayer`` via
      ``PredictionGTF.make_default_hmm``.
    * Chunk and one-hot encode DNA using ``GenomeSequences.get_flat_chunks``
      behind ``PredictionGTF.load_genome_data``.
    * Run the LSTM forward pass with ``PredictionGTF.lstm_prediction``.
    * Decode with the HMM using ``PredictionGTF.hmm_predictions_filtered``
      (the same early-CDS heuristic used by the CLI when `--hmm_filter` is on).
    """

    def __init__(
        self,
        genome: Dict[str, SimpleSeqRecord],
        seq_len: int = 180,
        batch_size: int = 2,
        strand: str = "+",
        softmask: bool = True,
        parallel_factor: int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.genome = genome
        self.strand = strand
        self.softmask = softmask

        # Build the same LSTM architecture used in training, but keep it
        # lightweight for demos (fewer filters/units).
        self.backbone = lstm_model(
            units=32,
            filter_size=16,
            kernel_size=9,
            numb_conv=2,
            numb_lstm=1,
            pool_size=1,
            output_size=15,
            softmasking=self.softmask,
        )

        # PredictionGTF handles batching, chunking, and HMM decoding.
        self.runner = PredictionGTF(
            genome=genome,
            seq_len=seq_len,
            batch_size=batch_size,
            softmask=self.softmask,
            hmm=True,
            strand=strand,
            parallel_factor=parallel_factor,
        )
        self.runner.hmm_factor = 1
        # Skip on-disk model loading; use the freshly built backbone instead.
        self.runner.lstm_model = self.backbone
        self.runner.adapted_batch_size = batch_size
        self.runner.gene_pred_hmm_layer = None
        self.runner.make_default_hmm(inp_size=self.backbone.output_shape[-1])

    def _load_chunks(self) -> Tuple[np.ndarray, List[List], int]:
        """One-hot encode and chunk the genome using ``PredictionGTF`` helpers."""

        fasta = self.runner.init_fasta(chunk_len=self.seq_len)
        seq_names = fasta.sequence_names
        f_chunk, coords, adapted_len = self.runner.load_genome_data(
            fasta_object=fasta,
            seq_names=seq_names,
            strand=self.strand,
            softmask=self.softmask,
        )
        # If the chunk length had to shrink (short contigs), keep batch size in sync.
        self.runner.adapt_batch_size(adapted_len)
        return f_chunk, coords, adapted_len

    def run(self) -> Tuple[np.ndarray, np.ndarray, List[List]]:
        """Execute the forward pass through LSTM logits and HMM labels.

        Returns a tuple of three elements:
        1. The raw nucleotide chunks (``f_chunk``) from ``load_genome_data``.
        2. The HMM label matrix from ``hmm_predictions_filtered``.
        3. The per-chunk coordinate metadata aligned with ``f_chunk``.
        """

        nuc_chunks, coords, _ = self._load_chunks()
        logits = self.runner.lstm_prediction(
            nuc_chunks,
            save=False,
            batch_size=self.runner.adapted_batch_size,
        )
        hmm_labels = self.runner.hmm_predictions_filtered(
            nuc_chunks,
            logits,
            save=False,
            batch_size=self.runner.adapted_batch_size,
        )
        return nuc_chunks, hmm_labels, coords



def demo_run() -> None:
    """Generate a toy DNA sequence and print HMM outputs.

    The synthetic contig is 180 bp long (divisible by the HMM requirements of
    2×9) to avoid chunk resizing. The output shows both the nucleotide chunk
    shape and the decoded HMM label IDs for quick debugging.
    """

    rng = np.random.default_rng(seed=7)
    # Bias toward exonic "ATG" pattern to trigger the CDS heuristic occasionally.
    bases = np.array(list("ACGT"))
    toy_sequence = "".join(rng.choice(bases, size=180))

    genome = {"toy_contig": SimpleSeqRecord(seq=toy_sequence)}
    pipeline = HmmOnlyPipeline(genome=genome, seq_len=180, batch_size=2)
    nuc_chunks, hmm_labels, coords = pipeline.run()

    print("Input chunk shape:", nuc_chunks.shape)
    print("HMM label shape:", hmm_labels.shape)
    print("Chunk coords:", coords)
    print("HMM labels (first chunk):", hmm_labels[0])


if __name__ == "__main__":
    demo_run()
