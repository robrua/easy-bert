"""
Runs a BERT model from the command line
"""
from typing import ContextManager
from contextlib import contextmanager
from pathlib import Path
import os

import numpy as np
import click

from .bert import Bert, _DEFAULT_MAX_SEQUENCE_LENGTH, _DEFAULT_PER_TOKEN_EMBEDDING
from . import __version__


@contextmanager
def _gpu(gpu: bool) -> ContextManager:
    """
    A contextmanager for controlling the visibility of CUDA devices.
    Allows for running on CPU or GPU on devices which support both

    Args:
        gpu (bool): whether to use the GPU
    """
    if gpu:
        yield
    else:
        try:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            visible_devices = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        yield
        if visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]


@contextmanager
def _errors_only(activate: bool) -> ContextManager:
    """
    A contextmanager for stopping TensorFlow from spamming the console with non-errors

    Args:
        activate (bool): whether to restrict TensorFlow logging to errors
    """
    if not activate:
        yield
    else:
        try:
            log_level = os.environ["TF_CPP_MIN_LOG_LEVEL"]
        except KeyError:
            log_level = None
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        yield
        if log_level:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = log_level
        else:
            del os.environ["TF_CPP_MIN_LOG_LEVEL"]


@click.group(help="Run a pretrained BERT model")
@click.version_option(version=__version__)
def _main() -> None:
    pass


_DEFAULT_ENCODING = "UTF-8"
_DEFAULT_HUB_MODEL = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
_DEFAULT_GPU = False
_DEFAULT_VERBOSE = False


@_main.command(name="embed", help="Gets BERT embeddings of provided data")
@click.option("-s", "--sequence", default=None, type=str, help="the sequence to embed")
@click.option("-i", "--input", default=None, type=str, help="the path to a .txt file containing sequences to embed, one per line")
@click.option("-e", "--encoding", default=_DEFAULT_ENCODING, help="the text encoding of the input file provided by (-i/--input)", show_default=True)
@click.option("-o", "--output", default=None, type=str, help="the path to put the resuling  [default: print the embeddings to console]", show_default=False)
@click.option("-m", "--model", default=None, type=str, help="the path to the TensorFlow saved model to use  [default: use a model from TensorFlow Hub (-h/--hub-model)]", show_default=False)
@click.option("-t/-p", "--tokens/--pooled", default=_DEFAULT_PER_TOKEN_EMBEDDING, help="whether to return per-token embeddings or pooled embeddings for the full sequences  [default: {}]".format("tokens" if _DEFAULT_PER_TOKEN_EMBEDDING else "pooled"), show_default=False)
@click.option("-h", "--hub-model", default=_DEFAULT_HUB_MODEL, help="the url to the TensorFlow Hub BERT model to use", show_default=True)
@click.option("-l", "--max-sequence-length", default=_DEFAULT_MAX_SEQUENCE_LENGTH, help="the max sequence length that a model initialized from TensorFlow Hub should allow, if one is being used", show_default=True)
@click.option("-g/-c", "--gpu/--cpu", default=_DEFAULT_GPU, help="whether to use the gpu  [default: {}]".format("gpu" if _DEFAULT_GPU else "cpu"), show_default=False)
@click.option("-v/-q", "--verbose/--quiet", default=_DEFAULT_VERBOSE, help="whether to log verbose TensorFlow output  [default: {}]".format("verbose" if _DEFAULT_VERBOSE else "quiet"), show_default=False)
def _embed(sequence: str = None,
           input: str = None,
           encoding: str = _DEFAULT_ENCODING,
           output: str = None,
           model: str = None,
           tokens: bool = _DEFAULT_PER_TOKEN_EMBEDDING,
           hub_model: str = _DEFAULT_HUB_MODEL,
           max_sequence_length: int = _DEFAULT_MAX_SEQUENCE_LENGTH,
           gpu: bool = _DEFAULT_GPU,
           verbose: bool = _DEFAULT_VERBOSE) -> None:
    # Check inputs
    if sequence is None and input is None:
        print("Error: Missing option \"-s\" / \"--sequence\" or \"-i\" / \"--input\". Please include a sequence or input file of sequences to embed.")
        exit(1)
    if sequence is not None and input is not None:
        print("Error: Redundant options \"-s\" / \"--sequence\" and \"-i\" / \"--input\". Only one of these options should be provided.")
        exit(1)

    # Get sequences to embed
    if sequence is not None:
        sequences = sequence
    elif input is not None:
        input = Path(input)
        with input.open("r", encoding=encoding) as in_file:
            sequences = [sequence.strip() for sequence in in_file]

    with _errors_only(not verbose), _gpu(gpu):
        # Load model
        if model is not None:
            bert = Bert.load(path=model)
        else:
            bert = Bert(tf_hub_url=hub_model, max_sequence_length=max_sequence_length)

        # Embed
        embeddings = bert.embed(sequences=sequences, per_token=tokens)

        # Output embeddings
        if output is not None:
            np.save(output, embeddings, allow_pickle=False)
        else:
            print(embeddings)


@_main.command(name="download", help="Downloads a TensorFlow Hub BERT model and converts it into a TensorFlow saved model")
@click.option("-m", "--model", required=True, type=str, help="the path to save the BERT model to")
@click.option("-h", "--hub-model", default=_DEFAULT_HUB_MODEL, help="the url to the TensorFlow Hub BERT model to use", show_default=True)
@click.option("-l", "--max-sequence-length", default=_DEFAULT_MAX_SEQUENCE_LENGTH, help="the max sequence length that the model should allow", show_default=True)
@click.option("-o/-s", "--overwrite/--safe", default=False, help="whether to overwrite the model directory if there's already a file or directory ther  [default: safe]", show_default=False)
@click.option("-g/-c", "--gpu/--cpu", default=_DEFAULT_GPU, help="whether to use the gpu  [default: {}]".format("gpu" if _DEFAULT_GPU else "cpu"), show_default=False)
@click.option("-v/-q", "--verbose/--quiet", default=_DEFAULT_VERBOSE, help="whether to log verbose TensorFlow output  [default: {}]".format("verbose" if _DEFAULT_VERBOSE else "quiet"), show_default=False)
def _download(model: str,
              hub_model: str = _DEFAULT_HUB_MODEL,
              max_sequence_length: int = _DEFAULT_MAX_SEQUENCE_LENGTH,
              overwrite: bool = False,
              gpu: bool = _DEFAULT_GPU,
              verbose: bool = _DEFAULT_VERBOSE) -> None:
    with _errors_only(not verbose), _gpu(gpu):
        bert = Bert(tf_hub_url=hub_model, max_sequence_length=max_sequence_length)
        bert.save(path=model, overwrite=overwrite)


if __name__ == "__main__":
    _main(prog_name="bert")
