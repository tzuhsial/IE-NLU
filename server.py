from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from flask import Flask
from flask import request, jsonify
from model import LSTMTagger
from reader import EditmeDatasetReader

app = Flask(__name__)

model = None


@app.route('/tag')
def tag():
    pass


if __name__ == "__main__":

    work_dir = 'work_dir'
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

    app.run(host='0.0.0.0', port=2004)
