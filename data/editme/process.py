"""Data related functions
Usage:
    data.py --csv=<file> --save=<file>

Options:
    -h --help       Show this screen.
    -v --version    Show version.
    --csv=<file>    CSV file path.
    --save=<file>   Save file path
"""
import csv
import io

from allennlp.data.tokenizers import WordTokenizer
from docopt import docopt
from tqdm import tqdm

tokenizer = WordTokenizer()

"""
Dataset reader for Joint intent and slot prediction tasks
    1370|150397|user5|edit the harsh lighting| [ier : [action(adjust) : edit] [attribute : the harsh lighting]]
    1447|2317387|user5|make the sky brighter| [ier : [action(adjust) : [attribute : brighter]]]
    1463|2317387|user5|the photo is too magnified| [comment : the photo is too magnified]
"""


def parse_annotation(string):
    """
    Annotate with IOB + intent
    """
    # ann: word -> tag pair
    # Handle unannotated case
    string = string.strip()
    if not is_bracket(string):
        if '[' in string and ']' in string:
            string = '[ier : ' + string + "]"
        else:
            string = '[comment : ' + string + "]"

    intent = None
    ann = {}

    queue = list()

    queue.append(string)
    while len(queue) > 0:
        # Parse bracket
        bracket = queue.pop(0)
        begin = bracket.find('[')
        end = bracket.rfind(']')
        key, value = bracket[begin+1:end].split(':', 1)
        key = key.strip()
        value = value.strip()

        # Check key value here
        if key == "comment":
            intent = key
        elif key.startswith('action'):
            # action(adjust) -> action_adjust
            action_type = key[7:-1]
            intent = action_type

        tokens = split_brackets(value)
        if any(is_bracket(token) for token in tokens):
            # If is token, then O tag, else throw back into queue
            for token in tokens:
                if is_bracket(token):
                    queue.append(token)
                else:
                    ann[token] = "O"
        else:
            # All the tokens are plain words, do BI tagging
            if key.startswith('action'):
                feature = "action"
            else:
                feature = key

            for idx, token in enumerate(tokens):
                if idx == 0:
                    label = "B-" + feature
                else:
                    label = "I-" + feature

                ann[token] = label

    return intent, ann


def split_brackets(string):
    """
    Horizontally split brackets and words 
    Returns:
        List[str] : List of strings, string can be a word or a bracket
    """
    tokens = []
    string = string.strip()
    begin = 0
    end = 0
    while begin < len(string):
        space_pos = string.find(" ", begin)
        open_pos = string.find("[", begin)

        if open_pos < 0 and space_pos < 0:
            end = len(string)
            spacy_tokens = tokenizer.tokenize(string[begin:end])
            tokens += [st.text for st in spacy_tokens]
        elif open_pos >= 0 and open_pos < space_pos:
            # Find valid closing bracket
            stack = list()
            for idx, char in enumerate(string[open_pos:], open_pos):
                if char == "[":
                    stack.append(char)
                elif char == "]":
                    stack.pop()
                    if len(stack) == 0:
                        close_pos = idx
                        break
            begin = open_pos
            end = close_pos+1
            token = string[begin:end]
            tokens.append(token)
        else:
            end = space_pos
            spacy_tokens = tokenizer.tokenize(string[begin:end])
            tokens += [st.text for st in spacy_tokens]
        begin = end + 1
    return tokens


def is_bracket(string):
    if '[' not in string and ']' not in string:
        return False
    # hahahah
    stack = list()
    for idx, char in enumerate(string):
        if char == "[":
            stack.append(char)
        elif char == "]":
            if len(stack) == 0:
                return False
            stack.pop()

            if idx != len(string)-1 and len(stack) == 0:
                return False
    return len(stack) == 0


def read_and_parse(csv_file):

    data = []
    tokenizer = WordTokenizer()
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin, delimiter="|")
        next(reader)  # Skip header
        for row in tqdm(reader):
            _, _, _, command, annotation = row

            # How to split this crazy ass thing
            tokens = [tok.text for tok in tokenizer.tokenize(command)]
            intent, ann = parse_annotation(annotation)

            labels = []
            for token in tokens:
                try:
                    if intent is None:
                        intent = "other"
                    label = ann[token]
                except:
                    print()
                    print("command", command)
                    print("annotation", annotation)
                    print("ann", ann)
                    import pdb
                    pdb.set_trace()
                labels.append(label)

            data.append((tokens, labels, intent))
    return data


def write_to_file(data, save_file):
    with open(save_file, 'w') as fout:
        fout.write("tags|intent\n")
        for tokens, labels, intent in tqdm(data):

            tags = []
            for token, label in zip(tokens, labels):
                tag = token + "###" + label
                tags.append(tag)
            tags = " ".join(tags)
            fout.write(tags + "|" + intent + "\n")


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    print("reading from {}".format(args["--csv"]))
    data = read_and_parse(args["--csv"])
    print("writing to {}".format(args["--save"]))
    write_to_file(data, args["--save"])
