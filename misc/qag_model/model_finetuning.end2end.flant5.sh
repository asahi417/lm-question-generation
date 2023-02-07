#########
# SQuAD #
#########
lmqg-train-search -d "lmqg/qag_squad" -m "google/flan-t5-small" -b 32 -g 2 4 -c "lmqg_output/flan-t5-small-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/flan-t5-small-squad-qag/best_model" -e "lmqg_output/flan-t5-small-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/flan-t5-small-squad-qag/best_model" -a "flan-t5-small-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "google/flan-t5-base"  -b 8 -g 8 16 -c "lmqg_output/flan-t5-base-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/flan-t5-base-squad-qag/best_model" -e "lmqg_output/flan-t5-base-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/flan-t5-base-squad-qag/best_model" -a "flan-t5-base-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "google/flan-t5-large" -b 16 -g 4 8 -c "lmqg_output/flan-t5-large-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/flan-t5-large-squad-qag/best_model" -e "lmqg_output/flan-t5-large-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/flan-t5-large-squad-qag/best_model" -a "flan-t5-large-squad-qag" -o "lmqg"
