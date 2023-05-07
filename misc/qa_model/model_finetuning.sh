## SQuAD
#lmqg-train-search -m "t5-small" -d "lmqg/qg_squad" -b 64 -g 1 2 -c "lmqg_output/t5-small-squad-qa" -i 'paragraph_question' -o 'answer'
#lmqg-eval -m "lmqg_output/t5-small-squad-qa/best_model" -e "lmqg_output/t5-small-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
#lmqg-eval-qa -m "lmqg_output/t5-small-squad-qa/best_model" -e "lmqg_output/t5-small-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
#lmqg-push-to-hf -m "lmqg_output/t5-small-squad-qa/best_model" -a "t5-small-squad-qa" -o "lmqg"
#
#lmqg-train-search -m "t5-base" -d "lmqg/qg_squad" -b 32 -g 2 4 -c "lmqg_output/t5-base-squad-qa" -i 'paragraph_question' -o 'answer'
#lmqg-eval -m "lmqg_output/t5-base-squad-qa/best_model" -e "lmqg_output/t5-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
#lmqg-eval-qa -m "lmqg_output/t5-base-squad-qa/best_model" -e "lmqg_output/t5-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
#lmqg-push-to-hf -m "lmqg_output/t5-base-squad-qa/best_model" -a "t5-base-squad-qa" -o "lmqg"
#
#lmqg-train-search -m "facebook/bart-base" -d "lmqg/qg_squad" -b 32 -g 2 4 -c "lmqg_output/bart-base-squad-qa" -i 'paragraph_question' -o 'answer'
#lmqg-eval -m "lmqg_output/bart-base-squad-qa/best_model" -e "lmqg_output/bart-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
#lmqg-eval-qa -m "lmqg_output/bart-base-squad-qa/best_model" -e "lmqg_output/bart-base-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
#lmqg-push-to-hf -m "lmqg_output/bart-base-squad-qa/best_model" -a "bart-base-squad-qa" -o "lmqg"
#
#lmqg-train-search -m "t5-large" -d "lmqg/qg_squad" -b 16 -g 4 8 -c "lmqg_output/t5-large-squad-qa" -i 'paragraph_question' -o 'answer'
#lmqg-eval -m "lmqg_output/t5-large-squad-qa/best_model" -e "lmqg_output/t5-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
#lmqg-eval-qa -m "lmqg_output/t5-large-squad-qa/best_model" -e "lmqg_output/t5-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
#lmqg-push-to-hf -m "lmqg_output/t5-large-squad-qa/best_model" -a "t5-large-squad-qa" -o "lmqg"
#
#lmqg-train-search -m "facebook/bart-large" -d "lmqg/qg_squad" -b 32 -g 2 4 -c "lmqg_output/bart-large-squad-qa" -i 'paragraph_question' -o 'answer'
#lmqg-eval -m "lmqg_output/bart-large-squad-qa/best_model" -e "lmqg_output/bart-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
#lmqg-eval-qa -m "lmqg_output/bart-large-squad-qa/best_model" -e "lmqg_output/bart-large-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
#lmqg-push-to-hf -m "lmqg_output/bart-large-squad-qa/best_model" -a "bart-large-squad-qa" -o "lmqg"
#

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
for LA in "ko" "de" "ru" "it" "es" "ja" "fr"
do
  lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/mt5-small-${LA}quad-qa/best_model" -e "lmqg_output/mt5-small-${LA}quad-qa/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/mt5-small-${LA}quad-qa/best_model" -e "lmqg_output/mt5-small-${LA}quad-qa/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/mt5-small-${LA}quad-qa/best_model" -a "mt5-small-${LA}quad-qa" -o "lmqg"
done

lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_squad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "en" --n-max-config 1 -b 16 -g 4 8 -c "lmqg_output/mt5-small-squad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval -m "lmqg_output/mt5-small-squad-qa/best_model" -e "lmqg_output/mt5-small-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/mt5-small-squad-qa/best_model" -e "lmqg_output/mt5-small-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/mt5-small-squad-qa/best_model" -a "mt5-small-squad-qa" -o "lmqg"


for LA in "ko" "de" "ru" "it" "es" "ja" "fr"
do
  lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mbart-large-cc25-${LA}quad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/mbart-large-cc25-${LA}quad-qa/best_model" -e "lmqg_output/mbart-large-cc25-${LA}quad-qa/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-${LA}quad-qa/best_model" -e "lmqg_output/mbart-large-cc25-${LA}quad-qa/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/mbart-large-cc25-${LA}quad-qa/best_model" -a "mbart-large-cc25-${LA}quad-qa" -o "lmqg"
done

lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_squad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "en" --n-max-config 1 -b 8 -g 8 -c "lmqg_output/mbart-large-cc25-squad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -e "lmqg_output/mbart-large-cc25-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -e "lmqg_output/mbart-large-cc25-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -a "mbart-large-cc25-squad-qa" -o "lmqg"

########################
# mT5: French/Japanese #
########################
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_jaquad" --lr 8e-06 6e-06 4e-06 2e-06 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "ja" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mt5-small-jaquad-qa_1" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_jaquad" --lr 8e-05 6e-05 4e-05 2e-05 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "ja" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mt5-small-jaquad-qa_2" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_jaquad" --lr 8e-04 6e-04 4e-04 2e-04 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "ja" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mt5-small-jaquad-qa_3" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage

lmqg-eval-qa -m "lmqg_output/mt5-small-jaquad-qa_1/best_model" -e "lmqg_output/mt5-small-jaquad-qa_1/best_model/eval" -d "lmqg/qg_jaquad" --language "ja"
lmqg-eval-qa -m "lmqg_output/mt5-small-jaquad-qa_2/best_model" -e "lmqg_output/mt5-small-jaquad-qa_2/best_model/eval" -d "lmqg/qg_jaquad" --language "ja"
lmqg-eval-qa -m "lmqg_output/mt5-small-jaquad-qa_3/best_model" -e "lmqg_output/mt5-small-jaquad-qa_3/best_model/eval" -d "lmqg/qg_jaquad" --language "ja"
lmqg-eval -m "lmqg_output/mt5-small-jaquad-qa_3/best_model" -e "lmqg_output/mt5-small-jaquad-qa_3/best_model/eval" -d "lmqg/qg_jaquad" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/mt5-small-jaquad-qa_3/best_model" -a "mt5-small-jaquad-qa" -o "lmqg"

lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_frquad" --lr 8e-06 6e-06 4e-06 2e-06 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "fr" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mt5-small-frquad-qa_1" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_frquad" --lr 8e-05 6e-05 4e-05 2e-05 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "fr" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mt5-small-frquad-qa_2" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-train-search -m "google/mt5-small" -d "lmqg/qg_frquad" --lr 8e-04 6e-04 4e-04 2e-04 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "fr" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/mt5-small-frquad-qa_3" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage

lmqg-eval-qa -m "lmqg_output/mt5-small-frquad-qa_1/best_model" -e "lmqg_output/mt5-small-frquad-qa_1/best_model/eval" -d "lmqg/qg_frquad" --language "ja"
lmqg-eval-qa -m "lmqg_output/mt5-small-frquad-qa_2/best_model" -e "lmqg_output/mt5-small-frquad-qa_2/best_model/eval" -d "lmqg/qg_frquad" --language "ja"
lmqg-eval-qa -m "lmqg_output/mt5-small-frquad-qa_3/best_model" -e "lmqg_output/mt5-small-frquad-qa_3/best_model/eval" -d "lmqg/qg_frquad" --language "ja"
lmqg-eval -m "lmqg_output/mt5-small-frquad-qa_3/best_model" -e "lmqg_output/mt5-small-frquad-qa_3/best_model/eval" -d "lmqg/qg_frquad" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/mt5-small-frquad-qa_3/best_model" -a "mt5-small-frquad-qa" -o "lmqg"

####################
# mBART QG: French #
####################
lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_frquad" --lr 8e-04 6e-04 4e-04 2e-04 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "fr" --n-max-config 2 -b 16 -g 4 -c "lmqg_output/mbart-large-cc25-frquad-qg" --low-cpu-mem-usage
lmqg-eval -m "lmqg_output/mbart-large-cc25-frquad-qg/best_model" -e "lmqg_output/mbart-large-cc25-frquad-qg/best_model/eval" --language "fr" -d "lmqg/qg_frquad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/mbart-large-cc25-frquad-qg/best_model" -a "mbart-large-cc25-frquad-qg" -o "lmqg"

#################
# mBART: French #
#################
lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_frquad" --lr 8e-04 6e-04 4e-04 2e-04 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "fr" --n-max-config 2 -b 32 -g 2 -c "lmqg_output/mbart-large-cc25-frquad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-frquad-qa/best_model" -e "lmqg_output/mbart-large-cc25-frquad-qa/best_model/eval" -d "lmqg/qg_frquad" --language "fr"
lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_frquad" --lr 8e-05 6e-05 4e-05 2e-05 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "fr" --n-max-config 2 -b 16 -g 4 -c "lmqg_output/mbart-large-cc25-frquad-qa-1" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-frquad-qa-1/best_model" -e "lmqg_output/mbart-large-cc25-frquad-qa-1/best_model/eval" -d "lmqg/qg_frquad" --language "fr"
lmqg-eval -m "lmqg_output/mbart-large-cc25-frquad-qa/best_model" -e "lmqg_output/mbart-large-cc25-frquad-qa/best_model/eval" -d "lmqg/qg_frquad" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/mbart-large-cc25-frquad-qa/best_model" -a "mbart-large-cc25-frquad-qa" -o "lmqg"

##################
# mBART: English #
##################
lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_squad" --lr 8e-04 6e-04 4e-04 2e-04 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "en" --n-max-config 2 -b 16 -g 4 -c "lmqg_output/mbart-large-cc25-squad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -e "lmqg_output/mbart-large-cc25-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_squad" --lr 8e-05 6e-05 4e-05 2e-05 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "en" --n-max-config 2 -b 16 -g 4 -c "lmqg_output/mbart-large-cc25-squad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -e "lmqg_output/mbart-large-cc25-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-train-search -m "facebook/mbart-large-cc25" -d "lmqg/qg_squad" --lr 8e-06 6e-06 4e-06 2e-06 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "en" --n-max-config 2 -b 32 -g 2 -c "lmqg_output/mbart-large-cc25-squad-qa" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval-qa -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -e "lmqg_output/mbart-large-cc25-squad-qa/best_model/eval" -d "lmqg/qg_squad" --language "en"

lmqg-eval -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -e "lmqg_output/mbart-large-cc25-squad-qa/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-push-to-hf -m "lmqg_output/mbart-large-cc25-squad-qa/best_model" -a "mbart-large-cc25-squad-qa" -o "lmqg"
