"""
Usage:
    editme.py train [options]
    editme.py test [options]

Options:
    -h --help               Show this screen.
    --cuda=<bool>           use GPU [default: False]
    --seed=<int>            Seed [default: 123]
    --train-file=<file>     Path to train file [default: ./data/training.txt]
    --valid-file=<file>     Path to dev file [default: ./data/validation.txt]
    --test-file=<file>      Path to test file [default: ./data/testing.txt]
    --min-count=<int>       Vocabulary frequency cutoff [default: 2]
    --embed-dim=<int>       Embedding dimension [default: 100]
    --hidden-dim=<int>      Hidden dimension [default: 100]
    --batch-size=<int>      Batch size [default: 32]
    --lr=<float>            Learning rate [default: 0.1]
    --max-epoch=<int>       Max number of epochs[default: 100]
    --patience=<int>        Patience [default: 10]
    --work-dir=<file>       Work directory [default: ./work_dir]

"""

from typing import Iterator, List, Dict

from docopt import docopt
import torch
import torch.optim as optim
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from reader import EditmeDatasetReader
from model import LSTMTagger


def train(args):
    # Build dataset
    print("Building dataset...")
    reader = EditmeDatasetReader()
    train_dataset = reader.read(cached_path(args["--train-file"]))
    validation_dataset = reader.read(cached_path(args["--valid-file"]))
    #validation_dataset = None

    min_count = {
        "tokens": int(args["--min-count"])
    }

    vocab = Vocabulary.from_instances(
        train_dataset+validation_dataset, min_count=min_count)
    # Build model
    print("Building model...")
    embed_dim = int(args["--embed-dim"])
    hidden_dim = int(args["--hidden-dim"])

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=embed_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(
        embed_dim, hidden_dim, batch_first=True))
    model = LSTMTagger(word_embeddings, lstm, vocab)

    # Initialize training
    optimizer = optim.SGD(model.parameters(), lr=float(args["--lr"]))
    iterator = BucketIterator(batch_size=int(args["--batch-size"]),
                              sorting_keys=[("sentence", "num_tokens")],
                              track_epoch=True)
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=int(args["--patience"]),
                      num_epochs=int(args["--max-epoch"]),
                      serialization_dir=args["--work-dir"])
    print("Start training...")
    trainer.train()
    import pdb
    pdb.set_trace()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']


def test(args):
    raise NotImplementedError


def main():
    args = docopt(__doc__)
    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed)

    if args['train']:
        train(args)
    elif args['decode']:
        test(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
