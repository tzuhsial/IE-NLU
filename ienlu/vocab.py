from collections import Counter
from itertools import chain

from docopt import docopt


class VocabEntry(object):
    def __init__(self):
        self.word2id = {}
        self.word2id['<pad>'] = self.pad_id = 0
        self.word2id['<unk>'] = self.unk_id = 1

        self._id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self._id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self._id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, indices):
        if len(indices) == 0:
            return []
        elif type(indices[0]) == list:
            return [self.indices2words(i) for i in indices]
        else:
            return [self.id2word(i) for i in indices]

    @classmethod
    def from_corpus(self, corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types:', len(word_freq),
              ', >= freq:', len(valid_words))

        top_k_words = sorted(
            valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, sentences, tags, intents, size, cutoff):

        self.word = VocabEntry.from_corpus(sentences, size, cutoff)
        self.tag = VocabEntry.from_corpus(tags, size, 0)
        self.intent = VocabEntry.from_corpus([intents], size, 0)

    def __repr__(self):
        return 'Vocab(word: {}, tag: {}, intent: {})'\
            .format(len(self.word), len(self.tag), len(self.intent))
