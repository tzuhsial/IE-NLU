"""Train script

Usage:
    server.py --work-dir=<file> [options]

Options:
    -h --help               Show this screen.
    --version               Show version.
    -d --debug              Debug mode
    -p --port=<int>         Port [default: 5000]
    --work-dir=<file>       Work dir that contains model
"""
import os

from docopt import docopt
from flask import Flask, jsonify, request

from ienlu.model import IOBTagger

app = Flask(__name__)

model = None


def load_model(opt):
    global model
    model_path = os.path.join(opt['--work-dir'], 'model.pt')
    print("loading model", model_path)
    model = IOBTagger.load(model_path)


@app.route("/nlu", methods=["POST"])
def tag(opt):
    args = request.json or request.form

    sent = args['sent']
    print("Sentence", sent)

    result = model.tag(sent)
    print("predicted", result)
    return jsonify(result)


if __name__ == '__main__':
    opt = docopt(__doc__, version='0.1')

    port = int(opt['--port'])
    debug = opt['--debug']

    load_model(opt)

    print("running on port", port)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )
