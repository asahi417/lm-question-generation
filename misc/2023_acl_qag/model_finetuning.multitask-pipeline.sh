# Multitask QAG Model
# train, eval QG, eval AE, eval AR (answer metric), eval QAG, push-to-hub

lmqg-train-search -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5-large-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-qg-ae/best_model" -a "t5-large-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "t5-base"  -b 32 -g 2 4 8 -c "lmqg_output/t5-base-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-qg-ae/best_model" -a "t5-base-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "t5-small" -b 64 -g 1 2 4 -c "lmqg_output/t5-small-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-qg-ae/best_model" -a "t5-small-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -b 64 -g 1 2 -c "lmqg_output/bart-large-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/bart-large-squad-qg-ae/best_model" -e "lmqg_output/bart-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/bart-large-squad-qg-ae/best_model" -e "lmqg_output/bart-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-large-squad-qg-ae/best_model" -e "lmqg_output/bart-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/bart-large-squad-qg-ae/best_model" -e "lmqg_output/bart-large-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/bart-large-squad-qg-ae/best_model" -a "bart-large-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "facebook/bart-base"  -b 32 -g 2 4 -c "lmqg_output/bart-base-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/bart-base-squad-qg-ae/best_model" -e "lmqg_output/bart-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/bart-base-squad-qg-ae/best_model" -e "lmqg_output/bart-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-base-squad-qg-ae/best_model" -e "lmqg_output/bart-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/bart-base-squad-qg-ae/best_model" -e "lmqg_output/bart-base-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/bart-base-squad-qg-ae/best_model" -a "bart-base-squad-qg-ae" -o "lmqg"


lmqg-train-search -m "google/flan-t5-large" -b 16 -g 4 8 -c "lmqg_output/flan-t5-large-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-large-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/flan-t5-large-squad-qg-ae/best_model" -a "flan-t5-large-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-base"  -b 32 -g 2 4 8 -c "lmqg_output/flan-t5-base-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-base-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/flan-t5-base-squad-qg-ae/best_model" -a "flan-t5-base-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-small" -b 64 -g 1 2 4 -c "lmqg_output/flan-t5-small-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -e "lmqg_output/flan-t5-small-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/flan-t5-small-squad-qg-ae/best_model" -a "flan-t5-small-squad-qg-ae" -o "lmqg"


# Multitask QAG Model for non-English
mlqg_multi () {
  MODEL_NAME=${1}
  MODEL_ALIAS=${2}
  BATCH=${3}
  GRAD=${4}
  for LA in "ja" "es" "ko" "it" "de" "ru" "fr"
  do
    lmqg-train-search --use-auth-token -c "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae" -d "lmqg/qg_${LA}quad" -m "${MODEL_ALIAS}" -b ${BATCH} -g ${GRAD} --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" -o 'question'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-eval-qa --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-eval-qag --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad"
    lmqg-push-to-hf -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -a "${MODEL_NAME}-${LA}quad-qg-ae" -o "lmqg"
  done
}

mlqg_multi "mt5-small" "google/mt5-small" "64" "1"
mlqg_multi "mt5-base" "google/mt5-base" "32" "2"
mlqg_multi "mbart-large-cc25" "facebook/mbart-large-cc25" "2" "32"

# Answer extraction only model for Pipeline QAG
lmqg-train-search -m "t5-large" -b 4 -g 16 32 -c "lmqg_output/t5-large-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-large-squad-ae/best_model" -e "lmqg_output/t5-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-ae/best_model" -e "lmqg_output/t5-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-ae/best_model" -a "t5-large-squad-ae" -o "lmqg"

lmqg-train-search -m "t5-base" -b 16 -g 4 8 -c "lmqg_output/t5-base-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-base-squad-ae/best_model" -e "lmqg_output/t5-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-ae/best_model" -e "lmqg_output/t5-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-ae/best_model" -a "t5-base-squad-ae" -o "lmqg"

lmqg-train-search -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5-small-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-small-squad-ae/best_model" -e "lmqg_output/t5-small-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-ae/best_model" -e "lmqg_output/t5-small-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-ae/best_model" -a "t5-small-squad-ae" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -b 32 -g 2 4 -c "lmqg_output/bart-large-squad-ae" -i 'paragraph_sentence' -o 'answer'
lmqg-eval -m "lmqg_output/bart-large-squad-ae/best_model" -e "lmqg_output/bart-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-large-squad-ae/best_model" -e "lmqg_output/bart-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-large-squad-ae/best_model" -a "bart-large-squad-ae" -o "lmqg"

lmqg-train-search -m "facebook/bart-base" -b 16 -g 4 8 -c "lmqg_output/bart-base-squad-ae" -i 'paragraph_sentence' -o 'answer'
lmqg-eval -m "lmqg_output/bart-base-squad-ae/best_model" -e "lmqg_output/bart-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-base-squad-ae/best_model" -e "lmqg_output/bart-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-base-squad-ae/best_model" -a "bart-base-squad-ae" -o "lmqg"

lmqg-train-search -m "google/flan-t5-large" -b 4 -g 16 32 -c "lmqg_output/flan-t5-large-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
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


mlqg_ae () {
  MODEL_NAME=${1}
  MODEL_ALIAS=${2}
  BATCH=${3}
  GRAD=${4}
  for LA in "ja" "es" "ko" "it" "de" "ru" "fr" "zh"
  do
    lmqg-train-search --use-auth-token -c "lmqg_output/${MODEL_NAME}-${LA}quad-ae" -d "lmqg/qg_${LA}quad" -m "${MODEL_ALIAS}" -b ${BATCH} -g ${GRAD} --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --language "${LA}" --n-max-config 1 -i 'paragraph_sentence' -o 'answer'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i 'paragraph_sentence' -o 'answer'
    lmqg-eval-qa --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-push-to-hf -m "lmqg_output/${MODEL_NAME}-${LA}quad-ae/best_model" -a "${MODEL_NAME}-${LA}quad-ae" -o "lmqg"
    rm -rf "lmqg_output/${MODEL_NAME}-${LA}quad-ae"
  done
}

mlqg_ae "mt5-small" "google/mt5-small" "32" "2"
mlqg_ae "mt5-base" "google/mt5-base" "16" "4"
mlqg_ae "mbart-large-cc25" "facebook/mbart-large-cc25" "8" "8"

# Evaluate pipeline QAG: QG + QA models
git clone "https://huggingface.co/lmqg/t5-small-squad-qg"
lmqg-eval-qag -m "t5-small-squad-qg" --model-ae "lmqg/t5-small-squad-ae" -e "t5-small-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "t5-small-squad-qg" -a "t5-small-squad-qg" -o "lmqg"

git clone "https://huggingface.co/lmqg/t5-base-squad-qg"
lmqg-eval-qag -m "t5-base-squad-qg" --model-ae "lmqg/t5-base-squad-ae" -e "t5-base-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "t5-base-squad-qg" -a "t5-base-squad-qg" -o "lmqg"

git clone "https://huggingface.co/lmqg/t5-large-squad-qg"
lmqg-eval-qag -m "t5-large-squad-qg" --model-ae "lmqg/t5-large-squad-ae" -e "t5-large-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "t5-large-squad-qg" -a "t5-large-squad-qg" -o "lmqg"

git clone "https://huggingface.co/lmqg/bart-base-squad-qg"
lmqg-eval-qag -m "bart-base-squad-qg" --model-ae "lmqg/bart-base-squad-ae" -e "bart-base-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "bart-base-squad-qg" -a "bart-base-squad-qg" -o "lmqg"

git clone "https://huggingface.co/lmqg/bart-large-squad-qg"
lmqg-eval-qag -m "bart-large-squad-qg" --model-ae "lmqg/bart-large-squad-ae" -e "bart-large-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "bart-large-squad-qg" -a "bart-large-squad-qg" -o "lmqg"


mlqg_pipeline_qag () {
  MODEL_NAME=${1}
  LA="zh"
  git clone "https://huggingface.co/lmqg/${MODEL_NAME}-${LA}quad-qg"
  for LA in "ja" "es" "ko" "it" "de" "ru" "fr" "zh"
  do
    git clone "https://huggingface.co/lmqg/${MODEL_NAME}-${LA}quad-qg"
    lmqg-eval-qag --use-auth-token -m "${MODEL_NAME}-${LA}quad-qg" --model-ae "lmqg/${MODEL_NAME}-${LA}quad-ae" -e "${MODEL_NAME}-${LA}quad-qg/eval_pipeline" -d "lmqg/qg_${LA}quad" --language "${LA}"
    lmqg-push-to-hf -m "${MODEL_NAME}-${LA}quad-qg" -a "${MODEL_NAME}-${LA}quad-qg" -o "lmqg"
  done
}

mlqg_pipeline_qag "mt5-small"
mlqg_pipeline_qag "mt5-base"
mlqg_pipeline_qag "mbart-large-cc25"
