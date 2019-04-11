"""Train script

Usage:
    ienlu.py vocab TRAIN_TAG_FILE VOCAB_OUT
    ienlu.py train --train=<file> --valid=<file> --vocab=<file> --work-dir=<file> [options]
    ienlu.py test --test=<file> --work-dir=<file> [options]
    ienlu.py terminal --work-dir=<file> [options]

Options:
    -h --help               Show this screen.
    --version               Show version.
    --vocab-size=<int>      Vocabulary size [default: 50000]
    --vocab-cutoff=<int>    Vocabulary cutoff [default: 2]
    --train=<file>          Train tag file
    --valid=<file>          Valid tag file
    --test=<file>           Test tag file
    --vocab=<file>          Vocab binary
    --cuda                  Use cuda
    --work-dir=<file>       Work directory to save everything
    --embed-size=<int>      Size of embedding [default: 64]
    --hidden-size=<int>     Hidden size [default: 128]
    --num-layers=<int>      Number of layers [default: 1]
    --bidirectional         Bidirectional or not
    --dropout=<float>       Dropout rate [default: 0.1]
    --lr=<float>            Learning rate [default: 0.1]
    --momentum=<float>      Momentum [default: 0.9]
    --num-epochs=<int>      Number of epochs [default: 100]
    --batch-size=<int>      Batch size [default: 32]
    --patience=<int>        Patience for early stopping [default: 5]
    --seed=<int>            Random seed [default: 8591]
"""
import os
import pickle

import numpy as np
import torch
from docopt import docopt
from tqdm import tqdm

from ienlu.evaluate import computeF1Score
from ienlu.model import IOBTagger
from ienlu.util import batch_iter, load_from_pickle, read_tag_file
from ienlu.vocab import Vocab


def make_vocab(opt):
    """ Creates vocabulary binary file
    """
    size = int(opt['--vocab-size'])
    cutoff = int(opt['--vocab-cutoff'])

    # Read sentences
    tag_file = opt['TRAIN_TAG_FILE']
    sentences, tags, intents = read_tag_file(tag_file)

    # Create vocab
    vocab = Vocab(sentences, tags, intents, size, cutoff)
    print(vocab)

    # Save vocab binary
    vocab_out = opt["VOCAB_OUT"]
    print("saving vocab to", vocab_out)
    with open(vocab_out, 'wb') as fout:
        pickle.dump(vocab, fout)


def train(opt):
    # Data Config
    train_data = list(zip(*read_tag_file(opt['--train'])))
    valid_data = list(zip(*read_tag_file(opt['--valid'])))
    test_data = list(zip(*read_tag_file(opt['--test'])))
    vocab = load_from_pickle(opt['--vocab'])

    print("train", len(train_data))
    print("valid", len(valid_data))
    print("test", len(test_data))
    print(vocab)

    # Model config
    use_cuda = opt['--cuda'] and torch.cuda.is_available()

    conf = {
        "embed_size": int(opt['--embed-size']),
        "hidden_size": int(opt['--hidden-size']),
        "num_layers": int(opt['--num-layers']),
        "bidirectional": opt['--bidirectional'],
        "dropout": float(opt['--dropout']),
        "vocab": vocab,
        "use_cuda": use_cuda
    }
    model = IOBTagger(conf)
    model.init_weights()

    # Training Config
    lr = float(opt['--lr'])
    momentum = float(opt['--momentum'])

    print("optimizer: SGD")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    num_epochs = int(opt['--num-epochs'])
    batch_size = int(opt['--batch-size'])
    patience = int(opt['--patience'])

    # Main Loop Here
    best_valid_loss = None
    wait_patience = 0

    work_dir = opt['--work-dir']
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    model_save_path = os.path.join(work_dir, 'model.pt')

    for epoch in range(1, num_epochs+1):
        print('epoch', epoch)

        # Train
        model.train()

        cum_tag_loss = 0
        cum_intent_loss = 0
        cum_words = 0
        cum_sents = 0

        for batch_sents, batch_tags, batch_intents in batch_iter(train_data, batch_size, shuffle=True):

            optimizer.zero_grad()

            ntokens = sum(1 for sent in batch_sents for word in sent)
            nsents = len(batch_sents)

            # Forward
            tags_losses, intent_losses = model(
                batch_sents, batch_tags, batch_intents)

            tags_loss = tags_losses.sum() / ntokens
            intent_loss = intent_losses.sum() / nsents

            # Backprop
            loss = tags_loss + intent_loss

            loss.backward()
            optimizer.step()

            cum_tag_loss += tags_loss.cpu().data.item() * ntokens
            cum_intent_loss += intent_loss.cpu().data.item() * nsents

            cum_words += ntokens
            cum_sents += nsents

        train_tag_loss = cum_tag_loss / cum_words
        train_intent_loss = cum_intent_loss / cum_sents
        print("[train] tag loss {:2f} intent loss {:2f}"
              .format(train_tag_loss, train_intent_loss))

        # Valid
        cum_tag_loss = 0
        cum_intent_loss = 0
        cum_words = 0
        cum_sents = 0

        with torch.no_grad():
            model.eval()

            for batch_sents, batch_tags, batch_intents in batch_iter(valid_data, batch_size):
                ntokens = sum(1 for sent in batch_sents for word in sent)
                nsents = len(batch_sents)

                # Forward
                tags_losses, intent_losses = model(
                    batch_sents, batch_tags, batch_intents)

                tags_loss = tags_losses.sum() / ntokens
                intent_loss = intent_losses.sum() / nsents

                cum_tag_loss += tags_loss.cpu().data.item() * ntokens
                cum_intent_loss += intent_loss.cpu().data.item() * nsents

                cum_words += ntokens
                cum_sents += nsents

        valid_tag_loss = cum_tag_loss / cum_words
        valid_intent_loss = cum_intent_loss / cum_sents
        print("[valid] tag loss {:2f} intent loss {:2f}"
              .format(valid_tag_loss, valid_intent_loss))

        if best_valid_loss is None or valid_tag_loss < best_valid_loss:
            best_valid_loss = valid_tag_loss
            model.save(model_save_path)
        else:
            wait_patience += 1

        if wait_patience >= patience:
            print(
                "valid tag loss not improved for {} epochs, early stopping".format(patience))
            break


def test(opt):
    # data
    test_data = list(zip(*read_tag_file(opt['--test'])))

    # model
    work_dir = opt['--work-dir']
    model_path = os.path.join(work_dir, 'model.pt')
    use_cuda = opt['--cuda'] and torch.cuda.is_available()
    model = IOBTagger.load(model_path, use_cuda)

    f1scores = []
    precisions = []
    recalls = []
    nintent = len(test_data)
    nintent_correct = 0
    
    ndata = len(test_data)
    nwrong = 0

    for sent, tags_true, intent_true in tqdm(test_data):

        tags_pred, intent_pred = model.predict(sent)

        f, p, r = computeF1Score(tags_pred, tags_true)

        if f < 100:
            nwrong += 1
            print("sent", sent)
            print("true", tags_true)
            print("pred", tags_pred)

        f1scores.append(f)
        precisions.append(p)
        recalls.append(r)

        nintent_correct += intent_pred == intent_true

    intent_acc = nintent_correct / nintent
    print("F1", np.mean(f1scores), "Precision", np.mean(precisions), "Recall",
            np.mean(recalls), "Intent", intent_acc)
    print("wrong ratio", nwrong/ndata)


def terminal(opt):
    # model
    work_dir = opt['--work-dir']
    model_path = os.path.join(work_dir, 'model.pt')
    use_cuda = opt['--cuda'] and torch.cuda.is_available()
    model = IOBTagger.load(model_path, use_cuda)

    while True:

        sent = input("Input: ")

        result = model.tag(sent)

        print("Predict:", result)


if __name__ == '__main__':
    opt = docopt(__doc__, version='0.1')
    seed = int(opt['--seed'])
    print("Using seed", seed)
    torch.manual_seed(seed)

    if opt["vocab"]:
        make_vocab(opt)
    elif opt["train"]:
        train(opt)
    elif opt["test"]:
        test(opt)
    elif opt["terminal"]:
        terminal(opt)
