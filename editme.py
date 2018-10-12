"""
Usage:
    editme.py train [options]
    editme.py serve [options]

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
    --num-layers=<int>      Number of layers in lstm [default: 1]
    --dropout-rate=<float>       Dropout rate [default: 0.0]
    --bidirectional=<bool>  Whether to use bidirectional encoding [default: False]
    --batch-size=<int>      Batch size [default: 32]
    --lr=<float>            Learning rate [default: 0.1]
    --max-epoch=<int>       Max number of epochs[default: 100]
    --patience=<int>        Patience [default: 10]
    --work-dir=<file>       Work directory [default: ./work_dir]
"""

import json
import os
from typing import Dict, Iterator, List

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import (PytorchSeq2SeqWrapper,
                                               Seq2SeqEncoder)
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.trainer import Trainer
from docopt import docopt

from model import LSTMTagger
from reader import EditmeDatasetReader, read_iter
from util import computeF1Score

from flask import Flask, request, jsonify


def train(args):
    # Build dataset
    print("Building dataset...")
    reader = EditmeDatasetReader()
    train_dataset = reader.read(cached_path(args["--train-file"]))
    validation_dataset = reader.read(cached_path(args["--valid-file"]))
    # test_dataset = reader.read(cached_path(args['--test-file']))

    min_count = {
        "tokens": int(args["--min-count"])
    }

    vocab = Vocabulary.from_instances(
        train_dataset+validation_dataset, min_count=min_count)
    vocab_save_dir = os.path.join(args["--work-dir"], "vocabulary")
    vocab.save_to_files(vocab_save_dir)

    # Build model
    print("Building model...")
    embed_dim = int(args["--embed-dim"])

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=embed_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm_args = {
        "input_size": embed_dim,
        "hidden_size": int(args["--hidden-dim"]),
        "num_layers": int(args["--num-layers"]),
        "batch_first": True,
        "dropout": float(args["--dropout-rate"]),
        "bidirectional": bool(args["--bidirectional"])
    }
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(**lstm_args))
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
                      num_serialized_models_to_keep=5,
                      serialization_dir=args["--work-dir"])
    print("Start training...")
    trainer.train()

    # Testing
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    print("Train set")
    train_metrics = evaluate(args["--train-file"], predictor, vocab)

    print("Validation set")
    valid_metrics = evaluate(args["--valid-file"], predictor, vocab)

    print("Test set")
    test_metrics = evaluate(args["--test-file"], predictor, vocab)

    save_json = os.path.join(args["--work-dir"], "results.json")

    results = {
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics
    }
    print("Savings results to", save_json)
    with open(save_json, 'w') as fout:
        json.dump(results, fout)


def evaluate(filepath, predictor, vocab):
    correct_slots = []
    predict_slots = []
    correct_intent = []
    predict_intent = []
    for sentence, tags, intent in read_iter(filepath):
        sentence = ' '.join(sentence)
        pred_output = predictor.predict(sentence)
        tag_logits = pred_output["tag_logits"]

        tag_ids = np.argmax(tag_logits, axis=-1)
        pred_tags = [vocab.get_token_from_index(
            i, 'IOB_labels') for i in tag_ids]

        intent_logits = pred_output["intent_logits"]
        intent_id = np.argmax(intent_logits, axis=-1)
        pred_intent = vocab.get_token_from_index(
            intent_id, 'intent_labels')

        correct_slots.append(tags)
        predict_slots.append(pred_tags)

        correct_intent.append(intent)
        predict_intent.append(pred_intent)

    f1, precision, recall = computeF1Score(correct_slots, predict_slots)
    print("F1:", f1, "Precision:", precision, "Recall:", recall)

    accuracy = np.mean(
        [a == b for a, b, in zip(correct_intent, predict_intent)])
    print("Intent accuracy: ", accuracy)

    metrics = {
        "F1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }
    return metrics


def serve(args):

    # Load Model
    work_dir = args["--work-dir"]
    vocab = Vocabulary.from_files(os.path.join(work_dir, 'vocabulary'))

    print("Building model...")
    embed_dim = int(args["--embed-dim"])

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=embed_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm_args = {
        "input_size": embed_dim,
        "hidden_size": int(args["--hidden-dim"]),
        "num_layers": int(args["--num-layers"]),
        "batch_first": True,
        "dropout": float(args["--dropout-rate"]),
        "bidirectional": bool(args["--bidirectional"])
    }
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(**lstm_args))
    model = LSTMTagger(word_embeddings, lstm, vocab)
    model.load_state_dict(torch.load(os.path.join(work_dir, 'best.th')))

    # Build predictor
    reader = EditmeDatasetReader()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    # Funny that we can do something like this

    app = Flask(__name__)

    @app.route("/tag", methods=["POST"])
    def tag():
        sentence = request.form.get("sentence", "").strip()
        if sentence == "":
            return jsonify({"sentence": "", "tags": "", "intent": ""})

        output = predictor.predict(sentence)
        tag_logits = output['tag_logits']
        intent_logits = output['intent_logits']

        tag_ids = np.argmax(tag_logits, axis=-1)
        pred_tags = [vocab.get_token_from_index(
            i, 'IOB_labels') for i in tag_ids]
        intent_id = np.argmax(intent_logits, axis=-1)
        pred_intent = vocab.get_token_from_index(
            intent_id, 'intent_labels')
        print("Sentence: ", sentence)
        print("Predicted tags: ", pred_tags)
        print("Predicted intent: ", pred_intent)
        print(pred_tags, pred_intent)

        return jsonify({"sentence": sentence, "tags": pred_tags, "intent": pred_intent})

    app.run('0.0.0.0', port=2005)


def main():
    args = docopt(__doc__)
    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed)
    if args['train']:
        train(args)
    elif args['serve']:
        serve(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
