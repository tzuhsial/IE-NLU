from typing import Iterator, List, Dict, Any

from allennlp.data import Instance
from allennlp.data.fields import LabelField, SequenceLabelField, TextField
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary


class EditmeDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None, intent: str = None) -> Instance:
        fields = {}
        sentence_field = TextField(tokens, self.token_indexers)

        fields["sentence"] = sentence_field

        if tags:
            label_field = SequenceLabelField(
                labels=tags, sequence_field=sentence_field, label_namespace="IOB_labels")
            fields["labels"] = label_field

        if intent:
            intent_field = LabelField(
                label=intent, label_namespace="intent_labels")
            fields["intent"] = intent_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            next(f)
            for line in f:
                pos, intent = line.strip().split('|')
                pairs = pos.split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags, intent)


def read_iter(file_path: str) -> Iterator[Any]:
    """
    An iterator that returns raw string for prediction
    """
    with open(file_path) as f:
        next(f)
        for line in f:
            pos, intent = line.strip().split('|')
            pairs = pos.split()
            sentence, tags = zip(*(pair.split("###") for pair in pairs))
            yield sentence, tags, intent


if __name__ == "__main__":
    reader = EditmeDatasetReader()
    train_dataset = reader.read("./data/training.txt")

    min_count = {
        "tokens": 2
    }
    vocab = Vocabulary.from_instances(train_dataset, min_count=min_count)
    print('min_count', min_count)
    print('vocab_size', vocab.get_vocab_size())
    import pdb
    pdb.set_trace()
