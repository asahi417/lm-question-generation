# AE
lmqg-train-search -m "google/switch-large-128" -b 4 -g 16 32 -c "lmqg_output/switch-large-128-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/switch-large-128-squad-ae/best_model" -e "lmqg_output/switch-large-128-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/switch-large-128-squad-ae/best_model" -e "lmqg_output/switch-large-128-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/switch-large-128-squad-ae/best_model" -a "switch-large-128-squad-ae" -o "lmqg"

lmqg-train-search -m "google/switch-base-128" -b 8 -g 8 16 -c "lmqg_output/switch-base-128-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/switch-base-128-squad-ae/best_model" -e "lmqg_output/switch-base-128-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/switch-base-128-squad-ae/best_model" -e "lmqg_output/switch-base-128-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/switch-base-128-squad-ae/best_model" -a "switch-base-128-squad-ae" -o "lmqg"

# Multi
lmqg-train-search -m "google/switch-large-128" -b 16 -g 4 8 -c "lmqg_output/switch-large-128-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/switch-large-128-squad-qg-ae/best_model" -e "lmqg_output/switch-large-128-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/switch-large-128-squad-qg-ae/best_model" -e "lmqg_output/switch-large-128-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/switch-large-128-squad-qg-ae/best_model" -e "lmqg_output/switch-large-128-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/switch-large-128-squad-qg-ae/best_model" -e "lmqg_output/switch-large-128-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/switch-large-128-squad-qg-ae/best_model" -a "switch-large-128-squad-qg-ae" -o "lmqg"

[RUN] lmqg-train-search -m "google/switch-base-128"  -b 8 -g 8 16 32 -c "lmqg_output/switch-base-128-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/switch-base-128-squad-qg-ae/best_model" -e "lmqg_output/switch-base-128-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/switch-base-128-squad-qg-ae/best_model" -e "lmqg_output/switch-base-128-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/switch-base-128-squad-qg-ae/best_model" -e "lmqg_output/switch-base-128-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/switch-base-128-squad-qg-ae/best_model" -e "lmqg_output/switch-base-128-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/switch-base-128-squad-qg-ae/best_model" -a "switch-base-128-squad-qg-ae" -o "lmqg"

# QG
lmqg-train-search -m "google/switch-large-128" -p qg -b 16 -g 4 8 16 32 -c "lmqg_output/switch-large-128-squad"
lmqg-eval -m "lmqg_output/switch-large-128-squad/best_model" -e "lmqg_output/switch-large-128-squad/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/switch-large-128-squad/best_model" -a "switch-large-128-squad-qg" -o "lmqg"

lmqg-train-search -m "google/switch-base-128" -p qg -b 8 -g 8 16 32 64 -c "lmqg_output/switch-base-128-squad"
lmqg-eval -m "lmqg_output/switch-base-128-squad/best_model" -e "lmqg_output/switch-base-128-squad/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/switch-base-128-squad/best_model" -a "switch-base-128-squad-qg" -o "lmqg"

# e2e
lmqg-train-search -d "lmqg/qag_squad" -m "google/switch-large-128" -b 2 -g 32 64 -c "lmqg_output/switch-large-128-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/switch-large-128-squad-qag/best_model" -e "lmqg_output/switch-large-128-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/switch-large-128-squad-qag/best_model" -a "switch-large-128-squad-qag" -o "lmqg"

[RUN] lmqg-train-search -d "lmqg/qag_squad" -m "google/switch-base-128" -b 8 -g 8 16 -c "lmqg_output/switch-base-128-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/switch-base-128-squad-qag/best_model" -e "lmqg_output/switch-base-128-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/switch-base-128-squad-qag/best_model" -a "switch-base-128-squad-qag" -o "lmqg"


# pipeline
git clone "https://huggingface.co/lmqg/flan-t5-small-squad-qg"
lmqg-eval-qag -m "flan-t5-small-squad-qg" --model-ae "lmqg/flan-t5-small-squad-ae" -e "flan-t5-small-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "flan-t5-small-squad-qg" -a "flan-t5-small-squad-qg" -o "lmqg"

