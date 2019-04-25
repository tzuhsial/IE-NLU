import re
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
from torch.nn import init

from .util import input_transpose


class IOBTagger(nn.Module):
    """
    Class module for IOB tagger
    Currently, implement all nn code in class
    TODO: separate model code 
    """

    def __init__(self, opt):
        super(IOBTagger, self).__init__()

        self.vocab = opt['vocab']

        embed_size = opt['embed_size']
        hidden_size = opt['hidden_size']
        num_layers = opt['num_layers']
        dropout = opt['dropout']
        bidirectional = opt['bidirectional']

        self.embed = nn.Embedding(
            num_embeddings=len(self.vocab.word),
            embedding_dim=embed_size
        )

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.tag_out = nn.Linear(
            in_features=hidden_size,
            out_features=len(self.vocab.tag)
        )

        self.intent_out = nn.Linear(
            in_features=hidden_size,
            out_features=len(self.vocab.intent)
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.tag.pad_id, reduction="none")

        self.use_cuda = opt['use_cuda']

    def init_weights(self, uniform_weight=0.1):
        """
        Initialize weights for all modules
        """
        for param in self.parameters():
            init.uniform_(param, -uniform_weight, uniform_weight)

    def __call__(self, sents, tags, intents):
        # token to index
        sents = self.vocab.word.words2indices(sents)
        tags = self.vocab.tag.words2indices(tags)
        intents = self.vocab.intent.words2indices(intents)

        # Convert sentence to tensor
        sents_ids = input_transpose(sents, self.vocab.word.pad_id)
        sents_tensor = torch.LongTensor(sents_ids)

        tags_ids = input_transpose(tags, self.vocab.tag.pad_id)
        tags_tensor = torch.LongTensor(tags_ids)

        intents_tensor = torch.LongTensor(intents)

        if self.use_cuda:
            sents_tensor = sents_tensor.cuda()
            tags_tensor = tags_tensor.cuda()
            intents_tensor = intents_tensor.cuda()

        # (seq_len, batch_size, embed_size)
        embed_x = self.embed(sents_tensor)
        # (seq_len, batch_size, hidden_size)
        rnn_out, _ = self.rnn(embed_x)

        # Tags
        # (seq_len, batch_size, tag_size)
        tags_scores = self.tag_out(rnn_out)

        # Intents
        # (batch_size, hidden_size)
        rnn_last_out = rnn_out[-1]
        # (batch_size, intent_size)
        intent_scores = self.intent_out(rnn_last_out)

        # Tag Loss
        tags_true = tags_tensor.permute(1, 0)
        tags_pred = tags_scores.permute(1, 2, 0)
        tags_losses = self.criterion(tags_pred, tags_true)

        # Intent loss
        intent_losses = self.criterion(intent_scores, intents_tensor)

        return tags_losses, intent_losses

    def preprocess(self, sent):
        """ Preprocess sentence into our tagging format
        """
        if isinstance(sent, str):
            tokens = word_tokenize(sent)
        tokens = sent

        print("hard coded preprocess <attribute> and <value> in model")

        words = []
        for token in word_tokenize(sent):
            if token in ["brightness", "contrast", "hue", "saturation", "lightness"]:
                word = "<attribute>"
            elif token.isdigit() or (token[0] in ["-", "+"] and token[1:].isdigit()):
                word = "<value>"
            else:
                word = token
            words.append(word)
        return words

    def postprocess(self, sent, tags_pred, tags_intent):

        obj = {
            "action": [],
            "refer": [],
            "attribute": [],
            "value": [],
            "intent": tags_intent
        }

        # For convenience

        sent.append("</s>")
        tags_pred.append("O")

        prev_tag = None
        prev_words = []
        for word, tag in zip(sent, tags_pred):

            if tag == "O" or tag.startswith("B"):

                if prev_words:
                    assert prev_tag != "O"

                    _, prev_tag_type = prev_tag.split('-')

                    phrase = " ".join(prev_words)

                    obj[prev_tag_type].append(phrase)

                    prev_words = []

                if tag.startswith("B"):
                    _, tag_type = tag.split("-")
                    prev_words.append(word)

            else:
                B_or_I, tag_type = tag.split('-')
                assert B_or_I == "I"

                prev_words.append(word)

            prev_tag = tag
        return obj

    def predict(self, sent):
        """ Predict tags for a single sentence
        Args:
            sentence (list): list of str
        Returns:
            tags (list): tags for every word
            intent (str): predicted intent 
        """
        if isinstance(sent, str):
            pp_sent = self.preprocess(sent)
        else:
            pp_sent = sent

        # token to index
        sents = self.vocab.word.words2indices([pp_sent])

        # Convert sentence to tensor
        sents_ids = input_transpose(sents, self.vocab.word.pad_id)
        sents_tensor = torch.LongTensor(sents_ids)

        if self.use_cuda:
            sents_tensor = sents_tensor.cuda()

        # (seq_len, batch_size, embed_size)
        embed_x = self.embed(sents_tensor)
        # (seq_len, batch_size, hidden_size)
        rnn_out, _ = self.rnn(embed_x)

        # Tags
        # (seq_len, batch_size, tag_size)
        tags_logits = self.tag_out(rnn_out)

        # Intents
        # (batch_size, hidden_size)
        rnn_last_out = rnn_out[-1]

        # (batch_size, intent_size)
        intent_logits = self.intent_out(rnn_last_out)

        # Post process to labels
        assert tags_logits.shape[1] == 1
        # Predict tags
        tags_ids = tags_logits.squeeze(1).argmax(-1).cpu().tolist()

        tags_pred = self.vocab.tag.indices2words(tags_ids)

        # Predict intent
        intent_id = intent_logits.argmax().cpu().data.item()
        intent_pred = self.vocab.intent.id2word(intent_id)

        return tags_pred, intent_pred

    def tag(self, sent):
        if isinstance(sent, str):
            pre_sent = self.preprocess(sent)
            sent = word_tokenize(sent)

        tags_pred, intent_pred = self.predict(pre_sent)
        result = self.postprocess(sent, tags_pred, intent_pred)
        return result

    @staticmethod
    def load(model_path: str, use_cuda=False):
        """
        Load a pre-trained model
        Returns:
            model: the loaded model
        """
        if use_cuda:
            model = torch.load(model_path)
        else:  # load from cpu
            model = torch.load(
                model_path, map_location=lambda storage, loc: storage)
            model.use_cuda = False
        model.rnn.flatten_parameters()
        return model

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self, path)
