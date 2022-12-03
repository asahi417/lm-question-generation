
lmqg-train-search -m "t5-small" -d "lmqg/qg_tweetqa" -b 64 -g 1 2 -c "lmqg_output/t5-small-tweetqa-question-answering" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-small-tweetqa-question-answering/best_model" -e "lmqg_output/t5-small-tweetqa-question-answering/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-small-tweetqa-question-answering/best_model" -a "t5-small-tweetqa-question-answering" -o "lmqg"

lmqg-train-search -m "t5-base" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/t5-base-tweetqa-question-answering" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-base-tweetqa-question-answering/best_model" -e "lmqg_output/t5-base-tweetqa-question-answering/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-base-tweetqa-question-answering/best_model" -a "t5-base-tweetqa-question-answering" -o "lmqg"

lmqg-train-search -m "facebook/bart-base" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/bart-base-tweetqa-question-answering" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/bart-base-tweetqa-question-answering/best_model" -e "lmqg_output/bart-base-tweetqa-question-answering/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-base-tweetqa-question-answering/best_model" -a "bart-base-tweetqa-question-answering" -o "lmqg"



lmqg-train-search -m "t5-large" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/t5-large-tweetqa-question-answering" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-large-tweetqa-question-answering/best_model" -e "lmqg_output/t5-large-tweetqa-question-answering/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-large-tweetqa-question-answering/best_model" -a "t5-large-tweetqa-question-answering" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/bart-large-tweetqa-question-answering" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/bart-large-tweetqa-question-answering/best_model" -e "lmqg_output/bart-large-tweetqa-question-answering/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-large-tweetqa-question-answering/best_model" -a "bart-large-tweetqa-question-answering" -o "lmqg"

