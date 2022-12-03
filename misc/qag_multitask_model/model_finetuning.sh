# Multitask QAG Model
evaluate () { # CKPT LANG DATA
  lmqg-eval -m "lmqg_output/${1}/best_model" -e "lmqg_output/${1}/best_model/eval" --language "${2}" -d "lmqg/${3}" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/${1}/best_model" -a "${1}" -o "lmqg"
}

lmqg-train-search -m t5-large -b 16 -g 4 8 -c lmqg_output/t5-large-squad-multitask -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
evaluate t5-large-squad-multitask en qg_squad

lmqg-train-search -m t5-base  -b 32 -g 2 4 8 -c lmqg_output/t5-base-squad-multitask -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
evaluate t5-base-squad-multitask en qg_squad

lmqg-train-search -m t5-small -b 64 -g 1 2 4 -c lmqg_output/t5-large-squad-multitask -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
evaluate t5-large-squad-multitask en qg_squad

# Multitask QAG Model for non-English
mlqg_answer () {
  MODEL_NAME=${1}
  MODEL_ALIAS=${2}
  BATCH=${3}
  GRAD=${4}
  for LA in "ja" "es" "ko" "it" "de" "ru" "fr"
  do
    lmqg-train-search -c "lmqg_output/${MODEL_NAME//_/-}-${LA}quad-multitask" -d lmqg/qg_${LA}quad -m "${MODEL_ALIAS}" -b ${BATCH} -g ${GRAD} --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
    evaluate "${MODEL_NAME//_/-}-${LA}quad-multitask" ${LA} "qg_${LA}quad"
  done
}

mlqg_answer "mt5_small" "google/mt5-small" "64" "1"
mlqg_answer "mt5_base" "google/mt5-base" "32" "2"

# Ablation: answer extraction only model
evaluate_ae_only () { # CKPT LANG DATA
  lmqg-eval -m "lmqg_output/${1}/best_model" -e "lmqg_output/${1}/best_model/eval" --language "${2}" -d "lmqg/${3}" -i "paragraph_sentence" -o 'answer' --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/${1}/best_model" -a "${1}" -o "lmqg"
}

lmqg-train-search -m t5-large -b 4 -g 16 32 -c lmqg_output/t5-large-squad-answer-extraction -i 'paragraph_sentence' -o 'answer' -p 'ae'
evaluate_ae_only t5-large-squad-answer-extraction en qg_squad

lmqg-train-search -m t5-base -b 16 -g 4 8 -c lmqg_output/t5-base-squad-answer-extraction -i 'paragraph_sentence' -o 'answer' -p 'ae'
evaluate_ae_only t5-base-squad-answer-extraction en qg_squad

lmqg-train-search -m t5-small -b 64 -g 1 2 -c lmqg_output/t5-small-squad-answer-extraction -i 'paragraph_sentence' -o 'answer' -p 'ae'
evaluate_ae_only t5-small-squad-answer-extraction en qg_squad

