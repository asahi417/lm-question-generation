""" Scoring function of a question and answer pair. """
from typing import List
from difflib import SequenceMatcher


def append_score(
        context: str,
        list_qa: List,
        min_length_ratio: float = 0.1):

    list_qa_new = []
    for qa in list_qa:
        q = qa['question']
        a = qa['answer']
        if len(q) / len(a) < min_length_ratio:
            continue
        list_qa_new.append({
            "question": q,
            "answer": a,
            "score": 100 * (1 - SequenceMatcher(None, context, q).find_longest_match(0, len(context), 0, len(q)).size / len(q))
        })
    list_qa_new = sorted(list_qa_new, key=lambda x: x['score'], reverse=True)
    return list_qa_new



