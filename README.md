### EditmeTagger

A Bidirectional LSTM IOB tagger trained on the [Editme corpus](http://www.manuvinakurike.com/papers/editme_lrec.pdf).  Implemented with [allennlp](https://allennlp.org/).

### Getting Started

First create a python 3.6 environment and install the necessary packages

```bash
# Installation
conda create -n editme python=3.6
conda activate editme
pip install -r requirements.txt
python -m spacy download en
```

A trained model is already saved in ```./work_dir```.

To run in terminal
```bash
python editme.py terminal
```

A server can be run with port 2004 as default
```bash
python edtime.py serve
```




