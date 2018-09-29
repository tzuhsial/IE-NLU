from typing import Iterator, List, Dict

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

import torch
import torch.nn.functional as F


class LSTMTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('IOB_labels'))
        self.hidden2intent = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                             out_features=vocab.get_vocab_size('intent_labels'))
        self.tag_accuracy = CategoricalAccuracy()
        self.intent_accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,
                intent: torch.Tensor = None, **kwargs) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        last_encoder_out = encoder_out[:, -1, :]
        intent_logits = self.hidden2intent(last_encoder_out)
        output = {"tag_logits": tag_logits, "intent_logits": intent_logits}

        if labels is not None:
            self.tag_accuracy(tag_logits, labels, mask)
            output["tag_loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, mask)

            self.intent_accuracy(intent_logits, intent)
            output["intent_loss"] = F.cross_entropy(intent_logits, intent)

            output["loss"] = output["tag_loss"] + output["intent_loss"]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"tag_accuracy": self.tag_accuracy.get_metric(reset),
                "intent_accuracy": self.intent_accuracy.get_metric(reset)}
