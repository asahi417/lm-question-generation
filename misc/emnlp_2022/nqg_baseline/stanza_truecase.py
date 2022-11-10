import stanza
from stanza.server import CoreNLPClient

stanza.install_corenlp()


def stanza_true_case(target_text_list):

    # call CoreNLP process
    with CoreNLPClient(
            annotators=['truecase'],
            timeout=30000,
            memory='6G') as client:
        output = []
        for target_text in target_text_list:
            ann = client.annotate(target_text)

            fixed_text = ''
            before = None
            for sentence in ann.sentence:
                for token in sentence.token:
                    assert token.word == target_text[token.beginChar:token.endChar]
                    if before is not None:
                        assert token.before == before
                    before = token.after
                    fixed_text += token.trueCaseText
                    fixed_text += token.after

            output.append(fixed_text)
        return output


if __name__ == '__main__':

    text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
    text_lower = text.lower()
    text_fixed = stanza_true_case([text_lower])
    print('Text original : {}'.format(text))
    print('Text lowercase: {}'.format(text_lower))
    print('Text truecase : {}'.format(text_fixed[0]))

