# SQuAD
lmqg-train-search -m "t5-small" -d "lmqg/qg_squad" -b 64 -g 1 2 -c "lmqg_output/t5-small-squad-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-small-squad-qa/best_model" -e "lmqg_output/t5-small-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-qa/best_model" -e "lmqg_output/t5-small-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-qa/best_model" -a "t5-small-squad-qa" -o "lmqg"

lmqg-train-search -m "t5-base" -d "lmqg/qg_squad" -b 32 -g 2 4 -c "lmqg_output/t5-base-squad-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-base-squad-qa/best_model" -e "lmqg_output/t5-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-qa/best_model" -e "lmqg_output/t5-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-qa/best_model" -a "t5-base-squad-qa" -o "lmqg"

lmqg-train-search -m "facebook/bart-base" -d "lmqg/qg_squad" -b 32 -g 2 4 -c "lmqg_output/bart-base-squad-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/bart-base-squad-qa/best_model" -e "lmqg_output/bart-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-base-squad-qa/best_model" -e "lmqg_output/bart-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/bart-base-squad-qa/best_model" -a "bart-base-squad-qa" -o "lmqg"

lmqg-train-search -m "t5-large" -d "lmqg/qg_squad" -b 16 -g 4 8 -c "lmqg_output/t5-large-squad-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-large-squad-qa/best_model" -e "lmqg_output/t5-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-qa/best_model" -e "lmqg_output/t5-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-qa/best_model" -a "t5-large-squad-qa" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -d "lmqg/qg_squad" -b 32 -g 2 4 -c "lmqg_output/bart-large-squad-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/bart-large-squad-qa/best_model" -e "lmqg_output/bart-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-large-squad-qa/best_model" -e "lmqg_output/bart-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/bart-large-squad-qa/best_model" -a "bart-large-squad-qa" -o "lmqg"


# TweetQA
lmqg-train-search -m "t5-small" -d "lmqg/qg_tweetqa" -b 64 -g 1 2 -c "lmqg_output/t5-small-tweetqa-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-small-tweetqa-qa/best_model" -e "lmqg_output/t5-small-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-tweetqa-qa/best_model" -e "lmqg_output/t5-small-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-small-tweetqa-qa/best_model" -a "t5-small-tweetqa-qa" -o "lmqg"

lmqg-train-search -m "t5-base" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/t5-base-tweetqa-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-base-tweetqa-qa/best_model" -e "lmqg_output/t5-base-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-tweetqa-qa/best_model" -e "lmqg_output/t5-base-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-base-tweetqa-qa/best_model" -a "t5-base-tweetqa-qa" -o "lmqg"

lmqg-train-search -m "facebook/bart-base" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/bart-base-tweetqa-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/bart-base-tweetqa-qa/best_model" -e "lmqg_output/bart-base-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-base-tweetqa-qa/best_model" -e "lmqg_output/bart-base-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" --language "en"
lmqg-push-to-hf -m "lmqg_output/bart-base-tweetqa-qa/best_model" -a "bart-base-tweetqa-qa" -o "lmqg"

lmqg-train-search -m "t5-large" -d "lmqg/qg_tweetqa" -b 16 -g 4 8 -c "lmqg_output/t5-large-tweetqa-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/t5-large-tweetqa-qa/best_model" -e "lmqg_output/t5-large-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-tweetqa-qa/best_model" -e "lmqg_output/t5-large-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-large-tweetqa-qa/best_model" -a "t5-large-tweetqa-qa" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -d "lmqg/qg_tweetqa" -b 32 -g 2 4 -c "lmqg_output/bart-large-tweetqa-qa" -i 'paragraph_question' -o 'answer'
lmqg-eval -m "lmqg_output/bart-large-tweetqa-qa/best_model" -e "lmqg_output/bart-large-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-large-tweetqa-qa/best_model" -e "lmqg_output/bart-large-tweetqa-qa/best_model/eval" -d "lmqg/qg_tweetqa" --language "en"
lmqg-push-to-hf -m "lmqg_output/bart-large-tweetqa-qa/best_model" -a "bart-large-tweetqa-qa" -o "lmqg"


# Multi-QuAD
## WARNING: Models are not fine-tuned properly.
LA="ko"  # worked
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage --language "${LA}"
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
LA="de"  # running
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
LA="ru"  # running
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
LA="it"  # running
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
LA="es"  # running
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
LA="ja"  # running
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
LA="fr"  # running
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage

lmqg-eval -m "lmqg_output/mt5-small-${LA}quad-qa/best_model" -e "lmqg_output/mt5-small-${LA}quad-qa/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/mt5-small-${LA}quad-qa/best_model" -e "lmqg_output/mt5-small-${LA}quad-qa/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
lmqg-push-to-hf -m "lmqg_output/mt5-small-${LA}quad-qa/best_model" -a "mt5-small-${LA}quad-qa" -o "lmqg"
