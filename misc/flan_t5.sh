# AE
lmqg-train-search -m "google/flan-t5-large" -b 4 -g 8 16 -c "lmqg_output/flan-t5-large-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/flan-t5-large-squad-ae/best_model" -e "lmqg_output/flan-t5-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/flan-t5-large-squad-ae/best_model" -e "lmqg_output/flan-t5-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/flan-t5-large-squad-ae/best_model" -a "flan-t5-large-squad-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-base" -b 16 -g 4 8 -c "lmqg_output/flan-t5-base-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/flan-t5-base-squad-ae/best_model" -e "lmqg_output/flan-t5-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/flan-t5-base-squad-ae/best_model" -e "lmqg_output/flan-t5-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/flan-t5-base-squad-ae/best_model" -a "flan-t5-base-squad-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-small" -b 64 -g 1 2 -c "lmqg_output/flan-t5-small-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/flan-t5-small-squad-ae/best_model" -e "lmqg_output/flan-t5-small-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/flan-t5-small-squad-ae/best_model" -e "lmqg_output/flan-t5-small-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/flan-t5-small-squad-ae/best_model" -a "flan-t5-small-squad-ae" -o "lmqg"

# Multi
lmqg-train-search -m "google/flan-t5-large" -b 8 -g 8 16 -c "lmqg_output/flan-t5-large-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-eval-qa -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -a "flan-t5-large-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-base"  -b 8 -g 8 16 32 -c "lmqg_output/flan-t5-base-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-eval-qa -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -a "flan-t5-base-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-small" -b 32 -g 2 4 8 -c "lmqg_output/flan-t5-small-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-eval-qa -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -a "flan-t5-small-squad-qg-ae" -o "lmqg"

# QG
lmqg-train-search -m "google/flan-t5-large" -p qg -b 2 -g 32 64 128 -c "lmqg_output/flan-t5-large-squad"
lmqg-eval -m "lmqg_output/flan-t5-large-squad/best_model" -e "lmqg_output/flan-t5-large-squad/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/flan-t5-large-squad/best_model" -a "flan-t5-large-squad-qg" -o "lmqg"

lmqg-train-search -m "google/flan-t5-base" -p qg -b 16 -g 4 8 16 32 -c "lmqg_output/flan-t5-base-squad"
lmqg-eval -m "lmqg_output/flan-t5-base-squad/best_model" -e "lmqg_output/flan-t5-base-squad/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/flan-t5-base-squad/best_model" -a "flan-t5-base-squad-qg" -o "lmqg"

lmqg-train-search -m "google/flan-t5-small" -p qg -b 16 -g 4 8 16 32 -c "lmqg_output/flan-t5-small-squad"
lmqg-eval -m "lmqg_output/flan-t5-small-squad/best_model" -e "lmqg_output/flan-t5-small-squad/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/flan-t5-small-squad/best_model" -a "flan-t5-small-squad-qg" -o "lmqg"


# e2e
lmqg-train-search -d "lmqg/qag_squad" -m "google/flan-t5-large" -b 2 -g 32 64 -c "lmqg_output/flan-t5-large-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/flan-t5-large-squad-qag/best_model" -e "lmqg_output/flan-t5-large-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/flan-t5-large-squad-qag/best_model" -a "flan-t5-large-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "google/flan-t5-base"  -b 8 -g 8 16 -c "lmqg_output/flan-t5-base-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/flan-t5-base-squad-qag/best_model" -e "lmqg_output/flan-t5-base-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/flan-t5-base-squad-qag/best_model" -a "flan-t5-base-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "google/flan-t5-small" -b 16 -g 4 8 -c "lmqg_output/flan-t5-small-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/flan-t5-small-squad-qag/best_model" -e "lmqg_output/flan-t5-small-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/flan-t5-small-squad-qag/best_model" -a "flan-t5-small-squad-qag" -o "lmqg"

# pipeline
git clone "https://huggingface.co/lmqg/flan-t5-large-squad-qg"
lmqg-eval-qag -m "flan-t5-large-squad-qg" --model-ae "lmqg/flan-t5-large-squad-ae" -e "flan-t5-large-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "flan-t5-large-squad-qg" -a "flan-t5-large-squad-qg" -o "lmqg"

git clone "https://huggingface.co/lmqg/flan-t5-base-squad-qg"
lmqg-eval-qag -m "flan-t5-base-squad-qg" --model-ae "lmqg/flan-t5-base-squad-ae" -e "flan-t5-base-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "flan-t5-base-squad-qg" -a "flan-t5-base-squad-qg" -o "lmqg"

git clone "https://huggingface.co/lmqg/flan-t5-small-squad-qg"
lmqg-eval-qag -m "flan-t5-small-squad-qg" --model-ae "lmqg/flan-t5-small-squad-ae" -e "flan-t5-small-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "flan-t5-small-squad-qg" -a "flan-t5-small-squad-qg" -o "lmqg"

