import re
from stanza_truecase import stanza_true_case

path_to_prediction = 'nonlm.sample.test.hyp.txt'
path_to_output = 'nonlm_fixed.sample.test.hyp.txt'
_tok_dict = {"(": "-lrb-", ")": "-rrb-", "[": "-lsb-", "]": "-rsb-", "{": "-lcb-", "}": "-rcb-"}


def remove_artifact(text_list):
    # Truecase
    text_list = stanza_true_case(text_list)
    output = []
    for text in text_list:
        # convert into the representation of the gold question
        for k, v in _tok_dict.items():
            text = text.replace(v, k)

        # quote: `` '' --> "" ""
        text = re.sub(r"``\s*([^']+)\s*''", r'"\1"', text)
        text = re.sub(r'"\s*([^"]+)"', r'"\1"', text[::-1])[::-1]

        # add half space before eg) What? --> What ?
        for s in ["'s", "?", ",", "'", "$", "%", "."]:
            text = re.sub(r'\s*\{}'.format(s), '{}'.format(s), text)

        output.append(text)
    return output


if __name__ == '__main__':

    # TEST
    test = [
        "where was saint denis reputedly located ?",
        "what percentage of dna is no less than 20 % ?",
        "what did the term `` scientific socialism '' refer to ?"
    ]
    test_output = remove_artifact(test)
    print("TEST")
    for a, b in zip(test, test_output):
        print('\t original: {}'.format(a))
        print('\t fixed   : {}'.format(b))

    # MAIN
    with open(path_to_prediction) as f_reader:
        with open(path_to_output, 'w') as f_writer:
            out = remove_artifact([t for t in f_reader.read().split('\n') if len(t) > 0])
            f_writer.write('\n'.join(out))

