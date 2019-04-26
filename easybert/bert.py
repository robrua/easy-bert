"""
easy-bert is a dead simple API for using Google's high quality BERT language model (https://github.com/google-research/bert).

Currently, easy-bert is focused on getting embeddings from pre-trained BERT models. Support for fine-tuning and pre-training will be added in the future,
as well as support for using easy-bert for other tasks besides getting embeddings.

You can use easy-bert with pretrained BERT models from TensorFlow Hub or from local models in the TensorFlow saved model format.


To create a BERT embedder from a TensowFlow Hub model, simply instantiate a Bert object with the target tf-hub URL:

from easybert import Bert
bert = Bert("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")

You can also load a local model in TensorFlow's saved model format using Bert.load:

from easybert import Bert
bert = Bert.load("/path/to/your/model/")

Once you have a BERT model loaded, you can get sequence embeddings using bert.embed:

x = bert.embed("A sequence")
y = bert.embed(["Multiple", "Sequences"])

If you want per-token embeddings, you can set per_token=True:

x = bert.embed("A sequence", per_token=True)
y = bert.embed(["Multiple", "Sequences"], per_token=True)

easy-bert returns BERT embeddings as numpy arrays


Every time you call bert.embed, a new TensorFlow session is created and used for the computation. If you're calling bert.embed a lot
sequentially, you can speed up your code by sharing a TensorFlow session among those calls using a with statement:

with bert:
    x = bert.embed("A sequence", per_token=True)
    y = bert.embed(["Multiple", "Sequences"], per_token=True)


You can save a BERT model using bert.save, then reload it later using Bert.load:

bert.save("/path/to/your/model/")
bert = Bert.load("/path/to/your/model/")
"""
from typing import Union, Iterable
from types import TracebackType
from pathlib import Path

from bert.tokenization import FullTokenizer
from bert import run_classifier
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


_DEFAULT_MAX_SEQUENCE_LENGTH = 128  # Max number of BERT tokens in a sequence
_DEFAULT_PER_TOKEN_EMBEDDING = False  # Whether to return per-token embeddings or pooled embeddings for the full sequences
_SOURCE_MODEL_ASSET_FILE = "source-model.txt"  # The asset file in the saved model used to store the source tf-hub model URL


class Bert(object):
    """
    A BERT model that can be used for generating high quality sentence and word embeddings easily (https://github.com/google-research/bert)

    Args:
        tf_hub_url (str): the URL to the TensorFlow Hub model to load
        max_sequence_length (int): the maximum number of BERT tokens allowed in an input sequence
    """
    def __init__(self, tf_hub_url: str, max_sequence_length: int = _DEFAULT_MAX_SEQUENCE_LENGTH) -> None:
        self._source_model = tf_hub_url
        self._graph = tf.Graph()
        self._session = None

        # Initialize the BERT model
        with tf.Session(graph=self._graph) as session:
            # Download module from tf-hub
            bert_module = hub.Module(tf_hub_url)

            # Get the tokenizer from the module
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            vocab_file, do_lower_case = session.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
            self._tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

            # Create symbolic input tensors as inputs to the model
            self._input_ids = tf.placeholder(name="input_ids", shape=(None, max_sequence_length), dtype=tf.int32)
            self._input_mask = tf.placeholder(name="input_mask", shape=(None, max_sequence_length), dtype=tf.int32)
            self._segment_ids = tf.placeholder(name="segment_ids", shape=(None, max_sequence_length), dtype=tf.int32)

            # Get the symbolic output tensors
            self._outputs = bert_module({
                "input_ids": self._input_ids,
                "input_mask": self._input_mask,
                "segment_ids": self._segment_ids
            }, signature="tokens", as_dict=True)

    def __enter__(self) -> None:
        # Start a session
        if self._session is None:
            self._session = tf.Session(graph=self._graph)
            self._session.__enter__()
            self._session.run(tf.global_variables_initializer())

    def __exit__(self, exc_type: type = None, exc_value: Exception = None, traceback: TracebackType = None) -> None:
        # Close an open session
        if self._session is not None:
            self._session.__exit__(exc_type, exc_value, traceback)
            self._session = None

    def embed(self, sequences: Union[str, Iterable[str]], per_token: bool = _DEFAULT_PER_TOKEN_EMBEDDING) -> np.ndarray:
        """
        Embeds a sequence or multiple sequences using the BERT model

        Args:
            sequences (Union[str, Iterable[str]]): the sequence(s) to embed
            per_token (bool): whether to produce an embedding per token or a pooled embedding for the whole sequence

        Returns:
            a numpy array with the embedding(s) of the sequence(s)
        """
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]

        # Convert sequnces into BERT input format
        input_examples = [run_classifier.InputExample(guid=None, text_a=sequence, text_b=None, label=0) for sequence in sequences]
        input_features = run_classifier.convert_examples_to_features(input_examples, [0], self._input_ids.shape[1], self._tokenizer)

        # Execute the computation graph on the inputs
        if self._session is not None:
            output = self._session.run(self._outputs["sequence_output" if per_token else "pooled_output"], feed_dict={
                self._input_ids: [sequence.input_ids for sequence in input_features],
                self._input_mask: [sequence.input_mask for sequence in input_features],
                self._segment_ids: [sequence.segment_ids for sequence in input_features]
            })
        else:
            with tf.Session(graph=self._graph) as session:
                session.run(tf.global_variables_initializer())

                output = session.run(self._outputs["sequence_output" if per_token else "pooled_output"], feed_dict={
                    self._input_ids: [sequence.input_ids for sequence in input_features],
                    self._input_mask: [sequence.input_mask for sequence in input_features],
                    self._segment_ids: [sequence.segment_ids for sequence in input_features]
                })

        if single_input:
            output = output.reshape(output.shape[1:])
        return output

    def save(self, path: Union[str, Path], overwrite: bool = True) -> None:
        """
        Saves the BERT model to a directory as a TensorFlow saved model

        Args:
            path (Union[str, Path]): the directory to save the model to
            overwrite (bool): whether to automatically overwrite the directory if it already exists
        """
        if isinstance(path, str):
            path = Path(path)

        if path.exists():
            if not overwrite:
                raise ValueError("Model path already exists and overwrite was set to False")
            _delete(path)

        if self._session is not None:
            tf.saved_model.simple_save(self._session, str(path), inputs={
                "input_ids": self._input_ids,
                "input_mask": self._input_mask,
                "segment_ids": self._segment_ids
            }, outputs=self._outputs)
        else:
            with tf.Session(graph=self._graph) as session:
                session.run(tf.global_variables_initializer())

                tf.saved_model.simple_save(session, str(path), inputs={
                    "input_ids": self._input_ids,
                    "input_mask": self._input_mask,
                    "segment_ids": self._segment_ids
                }, outputs=self._outputs)

        # Save needed information to get the tokenizer when reloading
        with path.joinpath("assets", _SOURCE_MODEL_ASSET_FILE).open("w", encoding="UTF-8") as out_file:
            out_file.write("{}\n".format(self._source_model))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Bert":
        """
        Loads a BERT model that has been saved to a directory as a TensorFlow saved model

        Args:
            path (Union[str, Path]): the directory that contains the model

        Returns:
            the saved BERT model
        """
        if isinstance(path, str):
            path = Path(path)

        bert = cls.__new__(cls)
        bert._graph = tf.Graph()
        bert._session = None

        # Load graph from disk
        with tf.Session(graph=bert._graph) as session:
            bundle = tf.saved_model.load(session, ["serve"], str(path))

        # Redownload source model tokenizer from tf-hub
        with path.joinpath("assets", _SOURCE_MODEL_ASSET_FILE).open("r", encoding="UTF-8") as in_file:
            tf_hub_url = in_file.readline().strip()

        with tf.Session() as session:
            # Download module from tf-hub
            bert_module = hub.Module(tf_hub_url)

            # Get the tokenizer from the module
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            vocab_file, do_lower_case = session.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
            bert._tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        # Initialize inputs/outputs for use in bert.embed
        bert._input_ids = bert._graph.get_tensor_by_name(bundle.signature_def["serving_default"].inputs["input_ids"].name)
        bert._input_mask = bert._graph.get_tensor_by_name(bundle.signature_def["serving_default"].inputs["input_mask"].name)
        bert._segment_ids = bert._graph.get_tensor_by_name(bundle.signature_def["serving_default"].inputs["segment_ids"].name)
        bert._outputs = {
            "pooled_output": bert._graph.get_tensor_by_name(bundle.signature_def["serving_default"].outputs["pooled_output"].name),
            "sequence_output": bert._graph.get_tensor_by_name(bundle.signature_def["serving_default"].outputs["sequence_output"].name)
        }

        return bert


def _delete(path: Path) -> None:
    """
    Recursively deletes a Path regardless of whether it's a file, empty directory, or non-empty directory

    Args:
        path (Path): the path to delete
    """
    if not path.is_dir():
        path.unlink()
    else:
        for subpath in path.iterdir():
            _delete(subpath)
        path.rmdir()
