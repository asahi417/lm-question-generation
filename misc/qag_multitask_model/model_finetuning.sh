# Multitask QAG Model
# train, eval QG, eval AE, eval AR (answer metric), eval QAG, push-to-hub

lmqg-train-search -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5-large-squad-multitask" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-large-squad-multitask/best_model" -e "lmqg_output/t5-large-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-large-squad-multitask/best_model" -e "lmqg_output/t5-large-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-multitask/best_model" -e "lmqg_output/t5-large-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-large-squad-multitask/best_model" -e "lmqg_output/t5-large-squad-multitask/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-multitask/best_model" -a "t5-large-squad-multitask" -o "lmqg"

lmqg-train-search -m "t5-base"  -b 32 -g 2 4 8 -c "lmqg_output/t5-base-squad-multitask" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-base-squad-multitask/best_model" -e "lmqg_output/t5-base-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-base-squad-multitask/best_model" -e "lmqg_output/t5-base-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-multitask/best_model" -e "lmqg_output/t5-base-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-base-squad-multitask/best_model" -e "lmqg_output/t5-base-squad-multitask/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-multitask/best_model" -a "t5-base-squad-multitask" -o "lmqg"

lmqg-train-search -m "t5-small" -b 64 -g 1 2 4 -c "lmqg_output/t5-small-squad-multitask" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-small-squad-multitask/best_model" -e "lmqg_output/t5-small-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-small-squad-multitask/best_model" -e "lmqg_output/t5-small-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-multitask/best_model" -e "lmqg_output/t5-small-squad-multitask/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-small-squad-multitask/best_model" -e "lmqg_output/t5-small-squad-multitask/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-multitask/best_model" -a "t5-small-squad-multitask" -o "lmqg"

# Multitask QAG Model for non-English
mlqg_answer () {
  MODEL_NAME=${1}
  MODEL_ALIAS=${2}
  BATCH=${3}
  GRAD=${4}
  for LA in "ja" "es" "ko" "it" "de" "ru" "fr"
  do
    lmqg-train-search --use-auth-token -c "lmqg_output/${MODEL_NAME}-${LA}quad-multitask" -d "lmqg/qg_${LA}quad" -m "${MODEL_ALIAS}" -b ${BATCH} -g ${GRAD} --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model/eval" --language "${LA}" -d "qg_${LA}quad" -i "paragraph_answer" -o 'question'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model/eval" --language "${LA}" -d "qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-eval-qa --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model/eval" --language "${LA}" -d "qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-eval-qag --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad"
    lmqg-push-to-hf -m "lmqg_output/${MODEL_NAME}-${LA}quad-multitask/best_model" -a "${MODEL_NAME}-${LA}quad-multitask" -o "lmqg"
  done
}

mlqg_answer "mt5-small" "google/mt5-small" "64" "1"
mlqg_answer "mt5-base" "google/mt5-base" "32" "2"

# Ablation: answer extraction only model
lmqg-train-search -m "t5-large" -b 4 -g 16 32 -c "lmqg_output/t5-large-squad-answer-extraction" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-large-squad-answer-extraction/best_model" -e "lmqg_output/t5-large-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-answer-extraction/best_model" -e "lmqg_output/t5-large-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-answer-extraction/best_model" -a "t5-large-squad-answer-extraction" -o "lmqg"

lmqg-train-search -m "t5-base" -b 16 -g 4 8 -c "lmqg_output/t5-base-squad-answer-extraction" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-base-squad-answer-extraction/best_model" -e "lmqg_output/t5-base-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-answer-extraction/best_model" -e "lmqg_output/t5-base-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-answer-extraction/best_model" -a "t5-base-squad-answer-extraction" -o "lmqg"

lmqg-train-search -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5-small-squad-answer-extraction" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-small-squad-answer-extraction/best_model" -e "lmqg_output/t5-small-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-answer-extraction/best_model" -e "lmqg_output/t5-small-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-answer-extraction/best_model" -a "t5-small-squad-answer-extraction" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -b 4 -g 16 32 -c "lmqg_output/bart-large-squad-answer-extraction" -i 'paragraph_sentence' -o 'answer'
lmqg-eval -m "lmqg_output/bart-large-squad-answer-extraction/best_model" -e "lmqg_output/bart-large-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-large-squad-answer-extraction/best_model" -e "lmqg_output/bart-large-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-large-squad-answer-extraction/best_model" -a "bart-large-squad-answer-extraction" -o "lmqg"

lmqg-train-search -m "facebook/bart-base" -b 16 -g 4 8 -c "lmqg_output/bart-base-squad-answer-extraction" -i 'paragraph_sentence' -o 'answer'
lmqg-eval -m "lmqg_output/bart-base-squad-answer-extraction/best_model" -e "lmqg_output/bart-base-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-base-squad-answer-extraction/best_model" -e "lmqg_output/bart-base-squad-answer-extraction/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-base-squad-answer-extraction/best_model" -a "bart-base-squad-answer-extraction" -o "lmqg"
