import math
import pickle
import sys
import numpy as np


def parse_row(line):
    """ Parse row into x, y information
    """
    line = line.strip()

    word_tags, intent = line.split('|')

    words = []
    tag = []
    for word_tag in word_tags.split():
        word, word_tag = word_tag.split("###")
        words.append(word)
        tag.append(word_tag)

    return words, tag, intent


def read_tag_file(file_path):
    """  Reads tag file
    Args:
        file_path (str)
    Returns:
        sentences (list)
        tags (list)
        intent (list)
    """
    sentences = []
    tags = []
    intents = []

    with open(file_path, 'r') as fin:
        next(fin)  # Skip header
        for line in fin.readlines():
            words, tag, intent = parse_row(line)

            sentences.append(words)
            tags.append(tag)
            intents.append(intent)

    return sentences, tags, intents


def load_from_pickle(file_path):
    with open(file_path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        sents, tags, intents = list(zip(*examples))

        yield sents, tags, intents


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) >
                        i else pad_token for k in range(batch_size)])

    return sents_t
