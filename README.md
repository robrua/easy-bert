[![MIT Licensed](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robrua/bert/blob/master/LICENSE.txt)

# easy-bert
easy-bert is a dead simple API for using Google's high quality BERT language model (https://github.com/google-research/bert).

Currently, easy-bert is focused on getting embeddings from pre-trained BERT models. Support for fine-tuning and pre-training will be added in the future, as well as support for using easy-bert for other tasks besides getting embeddings.

Java bindings for inference using BERT models are another goal for this project and should be available soon.

## Installation
easy-bert is available on [PyPI](https://pypi.org/project/easybert/). You can install with `pip install easybert` or `pip install git+https://github.com/robrua/easy-bert.git` if you want the very latest.

## Usage
You can use easy-bert with pretrained BERT models from TensorFlow Hub or from local models in the TensorFlow saved model format.

To create a BERT embedder from a TensowFlow Hub model, simply instantiate a Bert object with the target tf-hub URL:

```python
from easybert import bert
bert = Bert("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")
```

You can also load a local model in TensorFlow's saved model format using `Bert.load`:

```python
from easybert import bert
bert = Bert.load("/path/to/your/model/")
```

Once you have a BERT model loaded, you can get sequence embeddings using `bert.embed`:

```python
x = bert.embed("A sequence")
y = bert.embed(["Multiple", "Sequences"])
```

If you want per-token embeddings, you can set `per_token=True`:

```python
x = bert.embed("A sequence", per_token=True)
y = bert.embed(["Multiple", "Sequences"], per_token=True)
```

easy-bert returns BERT embeddings as numpy arrays


Every time you call `bert.embed`, a new TensorFlow session is created and used for the computation. If you're calling `bert.embed` a lot sequentially, you can speed up your code by sharing a TensorFlow session among those calls using a `with` statement:

```python
with bert:
    x = bert.embed("A sequence", per_token=True)
    y = bert.embed(["Multiple", "Sequences"], per_token=True)
```

You can save a BERT model using `bert.save`, then reload it later using `Bert.load`:

```python
bert.save("/path/to/your/model/")
bert = Bert.load("/path/to/your/model/")
```

# CLI
easy-bert also provides a CLI tool to conveniently do one-off embeddings of sequences with BERT. It can also convert a TensorFlow Hub model to a saved model.

Run `bert --help`, `bert embed --help` or `bert download --help` to get details about the CLI tool.
