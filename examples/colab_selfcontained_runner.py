# coding: utf-8
"""
Self-contained Tiberius inference snippet for Google Colab.

This file copies the minimal inference-only pieces from the Tiberius
repository (no ``import tiberius``) so it can be pasted into a Colab
cell. It supports mammalian models with optional filtering, the choice
between downloaded finetuned weights or random initialization, and
exports the HMM Viterbi labels to ``.npy`` along with a textual class
legend derived from Gabriel et al. 2024 (Bioinformatics 40(12):btae685).

Usage inside Colab (single cell):

```python
# 1) install deps
!pip install tensorflow==2.12 numpy biopython requests

# 2) paste the contents of this file into a cell and run it

# 3) configure and execute
results = run_colab_inference(
    fasta_paths=["/content/my_genome.fa"],  # list of FASTA paths
    output_npy="/content/hmm_labels.npy",
    download_weights=True,     # or False for random init
    apply_filtering=True,      # toggle CDS-prefiltering heuristic
    strand="+",                # or "-"
)
print("Saved to", results["npy_path"])
```

This script keeps the original logic for genome chunking, CNN/BiLSTM
forward pass, HMM decoding, and optional early-CDS filtering. It omits
GTF construction but reports chunk coordinates and label meanings.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import math
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Helper layers and kernels (unchanged logic from tiberius.models/gene_pred_hmm)
# ---------------------------------------------------------------------------

class Cast(KL.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)


def make_aggregation_matrix(k=1):
    A = np.zeros((15*k, 5))
    for i in range(k):
        A[0+15*i, 0] = 1
        A[1+15*i:4+15*i, 1] = 1
        A[4+15*i:7+15*i, 2] = 1
        A[7+15*i, 0] = 1
        A[8+15*i:11+15*i, 3] = 1
        A[11+15*i:14+15*i, 4] = 1
        A[14+15*i, 4] = 1
    return tf.constant(A, dtype=tf.float32)


def make_15_class_emission_kernel(k=1):
    # Simple per-nucleotide preferences for 15 states
    emission_kernel = np.zeros((4, 15*k))
    for i in range(4):
        emission_kernel[i, 1 + i] = 1
        emission_kernel[i, 4 + i] = 1
        emission_kernel[i, 8 + i] = 1
        emission_kernel[i, 11 + i] = 1
    emission_kernel[0, 7] = 1
    emission_kernel[3, 14] = 1
    return tf.constant(emission_kernel, dtype=tf.float32)


def make_5_class_emission_kernel(k=1):
    emission_kernel = np.zeros((4, 1+4*k, 5))
    emission_kernel[:, 0, 0] = 1
    for i in range(k):
        emission_kernel[:, 1+4*i:1+4*i+3, 1] = np.eye(4)
        emission_kernel[:, 1+4*i+3, 2+i] = [1, 0, 0, 0]
    return tf.constant(emission_kernel, dtype=tf.float32)

# ---------------------------------------------------------------------------
# Genome handling (from tiberius.genome_fasta)
# ---------------------------------------------------------------------------

one_hot_table = np.zeros((256, 6), dtype=np.int32)
one_hot_table[:, 4] = 1
for base, vec in {
    "A": [1, 0, 0, 0, 0, 0],
    "C": [0, 1, 0, 0, 0, 0],
    "G": [0, 0, 1, 0, 0, 0],
    "T": [0, 0, 0, 1, 0, 0],
    "a": [1, 0, 0, 0, 0, 1],
    "c": [0, 1, 0, 0, 0, 1],
    "g": [0, 0, 1, 0, 0, 1],
    "t": [0, 0, 0, 1, 0, 1],
}.items():
    one_hot_table[ord(base), :] = vec


class GenomeSequences:
    def __init__(self, fasta_file: str = "", genome=None, chunksize: int = 20000, overlap: int = 0, min_seq_len: int = 0):
        self.fasta_file = fasta_file
        self.genome = genome
        self.chunksize = chunksize
        self.overlap = overlap
        self.min_seq_len = min_seq_len
        self.sequences: List[str] = []
        self.sequence_names: List[str] = []
        self.one_hot_encoded: Dict[str, np.ndarray] = {}
        if self.genome:
            self.extract_seqarray()
        elif self.fasta_file:
            self.read_fasta()

    def extract_seqarray(self):
        for name, seqrec in self.genome.items():
            if len(seqrec.seq) < self.min_seq_len:
                continue
            self.sequences.append(str(seqrec.seq))
            self.sequence_names.append(name)

    def read_fasta(self):
        opener = gzip.open if self.fasta_file.endswith(".gz") else open
        with opener(self.fasta_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if len(record.seq) < self.min_seq_len:
                    continue
                self.sequence_names.append(record.id)
                self.sequences.append(str(record.seq))

    def encode_sequences(self, seq: Optional[Sequence[str]] = None):
        if not seq:
            seq = self.sequence_names
        for name in seq:
            sequence = self.sequences[self.sequence_names.index(name)]
            int_seq = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            self.one_hot_encoded[name] = one_hot_table[int_seq]

    def get_flat_chunks(self, sequence_names=None, strand="+", pad=True, adapt_chunksize=False, parallel_factor=None):
        chunk_coords = None
        chunksize = self.chunksize
        if not sequence_names:
            sequence_names = self.sequence_names
        sequences_i = [self.one_hot_encoded[i] for i in sequence_names]
        if adapt_chunksize:
            max_len = max(len(seq) for seq in sequences_i)
            if max_len < self.chunksize:
                chunksize = max_len
                if chunksize <= 2 * self.overlap:
                    chunksize = min(2 * self.overlap + 1, self.chunksize)
                if parallel_factor is None:
                    parallel_factor = 1
                if chunksize < 2 * parallel_factor:
                    chunksize = 2 * parallel_factor
                divisor = 2 * 9 * parallel_factor // math.gcd(18, parallel_factor)
                chunksize = divisor * (1 + (chunksize - 1) // divisor)
        chunks_one_hot = []
        chunk_coords = []
        for idx, seq in enumerate(sequences_i):
            length = len(seq)
            num_chunks = (length + chunksize - 1) // chunksize
            for i in range(num_chunks):
                start = i * chunksize
                end = min((i + 1) * chunksize, length)
                chunk = seq[start:end]
                if pad and len(chunk) < chunksize:
                    pad_arr = np.zeros((chunksize - len(chunk), seq.shape[1]), dtype=seq.dtype)
                    chunk = np.vstack([chunk, pad_arr])
                chunks_one_hot.append(chunk)
                chunk_coords.append([sequence_names[idx], start, end])
        if strand == "-":
            chunks_one_hot = [chunk[::-1, ::-1].copy() for chunk in chunks_one_hot]
            chunk_coords = [[n, length - end, length - start] for (n, start, end), length in zip(chunk_coords, [len(seq) for seq in sequences_i for _ in range((len(seq)+chunksize-1)//chunksize)])]
        return np.stack(chunks_one_hot), chunk_coords, chunksize

# ---------------------------------------------------------------------------
# Model backbone (from tiberius.models.lstm_model)
# ---------------------------------------------------------------------------


def lstm_model(units=200, filter_size=64, kernel_size=9, numb_conv=1, numb_lstm=3, dropout_rate=0, pool_size=10, stride=1, lstm_mask=False, clamsa=False, output_size=7, softmasking=True, residual_conv=False, clamsa_kernel=5, lru_layer=False, inp_size=None,):
    if inp_size is None:
        inp_size = 6 if softmasking else 5
    input_ = KL.Input(shape=(None, inp_size), name="input")
    x = input_
    if not softmasking:
        x = Cast()(x)
    if stride > 1:
        x = KL.AveragePooling1D(pool_size=stride, strides=stride)(x)
    convs = []
    x_ = x
    for i in range(numb_conv):
        x_ = KL.Conv1D(filter_size, kernel_size, padding="same", activation="relu", name=f"conv_{i}")(x_)
        convs.append(x_)
    x = KL.LayerNormalization()(x_)
    if residual_conv:
        x = KL.Concatenate()([x, x_])
    x = KL.Concatenate()([x, x_])
    if pool_size > 1:
        x = KL.Reshape((-1, pool_size * x.shape[-1]))(x)
    x = KL.Dense(units * 2)(x)
    for i in range(numb_lstm):
        x = KL.Bidirectional(KL.LSTM(units, return_sequences=True, dropout=dropout_rate), name=f"bi_lstm_{i}")(x)
    x = KL.Dense(pool_size * output_size, name="out")(x)
    x = KL.Reshape((-1, output_size), name="lstm_out")(x)
    x = KL.Activation("softmax", name="softmax_out")(x)
    return Model(inputs=input_, outputs=x)

# ---------------------------------------------------------------------------
# HMM cell (from tiberius.gene_pred_hmm*)
# ---------------------------------------------------------------------------

class HmmCell(KL.Layer):
    def __init__(self, emitter, transitioner, parallel_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.emitter = emitter
        self.transitioner = transitioner
        self.parallel_factor = parallel_factor

    @property
    def state_size(self):
        return self.transitioner.num_states

    @property
    def output_size(self):
        return self.transitioner.num_states

    def call(self, inputs, states):
        prev_log_probs = states[0]
        emissions = self.emitter(inputs)
        trans = self.transitioner()
        next_log_probs = tf.reduce_logsumexp(prev_log_probs[:, :, None] + trans + emissions[:, None, :], axis=1)
        return next_log_probs, [next_log_probs]


class GenePredHMMEmitter(KL.Layer):
    def __init__(self, emission_kernel, share_intron_pars=True, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.emission_kernel_init = emission_kernel
        self.share_intron_pars = share_intron_pars

    def build(self, input_shape):
        self.emission_kernel = self.add_weight(
            shape=self.emission_kernel_init.shape,
            initializer=tf.constant_initializer(self.emission_kernel_init.numpy()),
            trainable=True,
            name="emission_kernel",
        )

    def call(self, inputs):
        nuc = inputs[..., :4]
        logits = inputs[..., 4:]
        log_emit = tf.einsum("bi,ik->bk", nuc, self.emission_kernel)
        num_states = int(self.emission_kernel.shape[-1])
        if logits.shape[-1] != num_states:
            pad = num_states - int(logits.shape[-1])
            logits = tf.pad(logits, [[0, 0], [0, pad]])
        return log_emit + logits


class GenePredHMMTransitioner(KL.Layer):
    def __init__(self, num_states=15, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.num_states = num_states

    def build(self, input_shape):
        self.transition_kernel = self.add_weight(
            shape=(self.num_states, self.num_states),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
            name="transition_kernel",
        )

    def call(self, inputs=None):
        return tf.nn.log_softmax(self.transition_kernel, axis=-1)


class GenePredHMMLayer(KL.Layer):
    def __init__(self, parallel_factor=1, num_hmm=1, share_intron_pars=True, emission_kernel=None, hmm_reducer=None, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.parallel_factor = parallel_factor
        self.num_hmm = num_hmm
        self.share_intron_pars = share_intron_pars
        self.hmm_reducer = hmm_reducer
        self.emission_kernel = emission_kernel if emission_kernel is not None else make_15_class_emission_kernel(k=num_hmm)

    def build(self, input_shape):
        self.emitter = GenePredHMMEmitter(self.emission_kernel, share_intron_pars=self.share_intron_pars, name="emitter")
        self.transitioner = GenePredHMMTransitioner(num_states=self.emission_kernel.shape[-1], name="transitioner")
        self.cell = HmmCell(self.emitter, self.transitioner, parallel_factor=self.parallel_factor)

    def call(self, inputs):
        nuc, logits = inputs
        emit_inp = tf.concat([nuc[..., :4], logits], axis=-1)
        rnn = KL.RNN(self.cell, return_sequences=True)
        log_probs = rnn(emit_inp)
        if self.hmm_reducer is not None:
            log_probs = tf.einsum("btk,km->btm", log_probs, self.hmm_reducer)
        return tf.argmax(log_probs, axis=-1)

    def predict_vit(self, nuc, logits, batch_size=8):
        preds = []
        for i in range(0, nuc.shape[0], batch_size):
            preds.append(self((nuc[i:i+batch_size], logits[i:i+batch_size])).numpy())
        return np.vstack(preds)

# ---------------------------------------------------------------------------
# Prediction wrapper (subset of tiberius.eval_model_class.PredictionGTF)
# ---------------------------------------------------------------------------

class PredictionGTF:
    def __init__(self, genome: Dict[str, SeqIO.SeqRecord], seq_len=500004, batch_size=64, softmask=True, strand="+", parallel_factor=1):
        self.genome = genome
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.adapted_batch_size = batch_size
        self.softmask = softmask
        self.strand = strand
        self.parallel_factor = parallel_factor
        self.hmm_factor = None
        self.gene_pred_hmm_layer: Optional[GenePredHMMLayer] = None
        self.lstm_model: Optional[Model] = None
        self.lstm_pred = None

    def make_default_hmm(self, inp_size):
        emission_kernel = make_15_class_emission_kernel(k=1)
        self.gene_pred_hmm_layer = GenePredHMMLayer(
            num_hmm=1,
            share_intron_pars=True,
            emission_kernel=emission_kernel,
            hmm_reducer=make_aggregation_matrix(k=1),
            parallel_factor=self.parallel_factor,
            name="gene_pred_hmm_layer",
        )
        # Build layer to materialize weights
        dummy_nuc = tf.zeros((1, self.seq_len, 5))
        dummy_logits = tf.zeros((1, self.seq_len, inp_size))
        _ = self.gene_pred_hmm_layer((dummy_nuc, dummy_logits))

    def adapt_batch_size(self, adapted_chunksize):
        self.adapted_batch_size = max(1, 2 ** int(np.log2(self.batch_size * self.seq_len // adapted_chunksize)))

    def init_fasta(self, chunk_len=None):
        return GenomeSequences(genome=self.genome, chunksize=chunk_len or self.seq_len)

    def load_genome_data(self, fasta_object: GenomeSequences, seq_names, strand="+", softmask=True):
        fasta_object.encode_sequences(seq_names)
        f_chunk, coords, adapted_len = fasta_object.get_flat_chunks(sequence_names=seq_names, strand=strand, adapt_chunksize=True, parallel_factor=self.parallel_factor)
        if not softmask:
            f_chunk = f_chunk[:, :, :5]
        return f_chunk, coords, adapted_len

    def lstm_prediction(self, f_chunk, batch_size=8):
        if self.lstm_model is None:
            raise RuntimeError("LSTM model not loaded")
        return self.lstm_model.predict(f_chunk, batch_size=batch_size)

    def hmm_predictions(self, f_chunk, lstm_pred, batch_size=8):
        if self.gene_pred_hmm_layer is None:
            raise RuntimeError("HMM layer missing")
        nuc = f_chunk.astype(np.float32)
        logits = lstm_pred.astype(np.float32)
        return self.gene_pred_hmm_layer.predict_vit(nuc, logits, batch_size=batch_size)

    def hmm_predictions_filtered(self, f_chunk, lstm_pred, batch_size=8):
        # same heuristic as original: run full hmm
        return self.hmm_predictions(f_chunk, lstm_pred, batch_size=batch_size)

# ---------------------------------------------------------------------------
# Weights handling and end-to-end runner
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "weights_url": "https://zenodo.org/record/10915840/files/hg38_softmask_model.tgz?download=1",
    "softmasking": True,
    "output_size": 7,
    "units": 200,
    "filter_size": 64,
    "kernel_size": 9,
    "numb_conv": 1,
    "numb_lstm": 3,
    "pool_size": 10,
    "stride": 1,
}

LABEL_DOC = {
    0: "Intergenic/background",
    1: "Intron phase 0",
    2: "Intron phase 1",
    3: "Intron phase 2",
    4: "Coding exon phase 0",
    5: "Coding exon phase 1",
    6: "Coding exon phase 2",
    7: "Start codon state",
    8: "Exon-intron boundary phase 0",
    9: "Exon-intron boundary phase 1",
    10: "Exon-intron boundary phase 2",
    11: "Intron-exon boundary phase 0",
    12: "Intron-exon boundary phase 1",
    13: "Intron-exon boundary phase 2",
    14: "Stop codon state",
}


def download_weights(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    archive = os.path.join(dest_dir, "hg38_softmask_model.tgz")
    if not os.path.exists(archive):
        with requests.get(MODEL_CONFIG["weights_url"], stream=True) as r:
            r.raise_for_status()
            with open(archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    import tarfile
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir)
    return os.path.join(dest_dir, "hg38_softmask_model")


def load_or_init_lstm(weights_path: Optional[str], softmask=True) -> Model:
    cfg = MODEL_CONFIG
    model = lstm_model(
        units=cfg["units"],
        filter_size=cfg["filter_size"],
        kernel_size=cfg["kernel_size"],
        numb_conv=cfg["numb_conv"],
        numb_lstm=cfg["numb_lstm"],
        pool_size=cfg["pool_size"],
        stride=cfg["stride"],
        output_size=cfg["output_size"],
        softmasking=softmask,
    )
    if weights_path:
        model.load_weights(os.path.join(weights_path, "weights.h5"))
    return model


def run_colab_inference(
    fasta_paths: Sequence[str],
    output_npy: str,
    download_weights: bool = True,
    apply_filtering: bool = True,
    strand: str = "+",
    seq_len: int = 1800,
    batch_size: int = 4,
):
    genome = {}
    for fp in fasta_paths:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                genome[record.id] = record
    weights_dir = None
    if download_weights:
        cache_dir = tempfile.mkdtemp(prefix="tiberius_weights_")
        weights_dir = download_weights(cache_dir)
    lstm = load_or_init_lstm(weights_dir if download_weights else None, softmask=True)
    runner = PredictionGTF(genome=genome, seq_len=seq_len, batch_size=batch_size, softmask=True, strand=strand, parallel_factor=1)
    runner.lstm_model = lstm
    runner.make_default_hmm(inp_size=lstm.output_shape[-1])
    fasta = runner.init_fasta(chunk_len=seq_len)
    seq_names = fasta.sequence_names
    f_chunk, coords, adapted_len = runner.load_genome_data(fasta, seq_names=seq_names, strand=strand, softmask=True)
    runner.adapt_batch_size(adapted_len)
    lstm_pred = runner.lstm_prediction(f_chunk, batch_size=runner.adapted_batch_size)
    labels = runner.hmm_predictions_filtered(f_chunk, lstm_pred, batch_size=runner.adapted_batch_size) if apply_filtering else runner.hmm_predictions(f_chunk, lstm_pred, batch_size=runner.adapted_batch_size)
    np.save(output_npy, labels.astype(np.int32))
    return {"labels": labels, "coords": coords, "npy_path": output_npy, "label_doc": LABEL_DOC}


if __name__ == "__main__":
    toy_fasta = os.path.join(tempfile.gettempdir(), "toy.fa")
    with open(toy_fasta, "w") as f:
        f.write(">toy\n" + "ATG" * 60 + "\n")
    results = run_colab_inference(
        fasta_paths=[toy_fasta],
        output_npy=os.path.join(tempfile.gettempdir(), "hmm_labels.npy"),
        download_weights=False,
        apply_filtering=True,
        seq_len=180,
    )
    print("Labels shape:", results["labels"].shape)
    print("Coords:", results["coords"])
    print("Saved legend:")
    for k, v in results["label_doc"].items():
        print(f"  {k}: {v}")
    print("NPY path:", results["npy_path"])
