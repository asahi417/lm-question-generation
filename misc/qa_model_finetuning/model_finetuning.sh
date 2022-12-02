MODEL="t5-small-tweetqa-question-answering"
DATA="lmqg/qg_tweetqa"
lmqg-train-search -m "t5-small" -d "${DATA}" -b 64 -g 1 2 -c "lmqg_output/${MODEL}" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/${MODEL}/best_model" -e "lmqg_output/${MODEL}/best_model/eval" -d "${DATA}" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/${MODEL}/best_model" -a "${MODEL}" -o "lmqg"
