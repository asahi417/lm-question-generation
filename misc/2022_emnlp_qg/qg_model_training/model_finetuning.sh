# ROBUST FINETUNING: Finetuning language model on question generation with our three-step hyperparameter search.
evaluate () { # CKPT LANG DATA ALIAS
  lmqg-eval -m "lmqg_output/${1}/best_model" -e "lmqg_output/${1}/best_model/eval" --language "${2}" -d "lmqg/${3}" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/${1}/best_model" -a "${4}" -o "lmqg"
  }

#########
# SQUAD #
#########
# QG
lmqg-train-search -m t5-large -p qg -b 16 -g 4 8 16 32 -c lmqg_output/t5_large_squad
evaluate t5_large_squad en qg_squad t5-large-squad-qg
lmqg-train-search -m t5-base -p qg -b 16 -g 4 8 16 32 -c lmqg_output/t5_base_squad
evaluate t5_base_squad en qg_squad t5-base-squad-qg
lmqg-train-search -m t5-small -p qg -b 64 -g 1 2 4 8 -c lmqg_output/t5_small_squad
evaluate t5_small_squad en qg_squad t5-small-squad-qg
lmqg-train-search -m facebook/bart-large -b 32 -g 2 4 8 16 -c lmqg_output/bart_large_squad
evaluate bart_large_squad en qg_squad bart-large-squad-qg
lmqg-train-search -m facebook/bart-base -b 32 -g 2 4 8 16 -c lmqg_output/bart_base_squad
evaluate bart_base_squad en qg_squad bart-base-squad-qg


lmqg-train-search -m google/flan-t5-large -p qg -b 16 -g 4 8 16 32 -c lmqg_output/flan_t5_large_squad
evaluate flan_t5_large_squad en qg_squad flan-t5-large-squad-qg
lmqg-train-search -m google/flan-t5-base -p qg -b 16 -g 4 8 16 32 -c lmqg_output/flan_t5_base_squad
evaluate flan_t5_base_squad en qg_squad flan-t5-base-squad-qg
lmqg-train-search -m google/flan-t5-small -p qg -b 64 -g 1 2 4 8 -c lmqg_output/flan_t5_small_squad
evaluate flan_t5_small_squad en qg_squad flan-t5-small-squad-qg


####################
# MultilingualQUAD #
####################
mlqg () {
  MODEL_NAME=${1}
  MODEL_ALIAS=${2}
  BATCH=${3}
  if [ "${BATCH}" = "64" ]
  then
    GRAD=1
  elif [ "${BATCH}" = "32" ]
  then
    GRAD=2
  elif [ "${BATCH}" = "16" ]
  then
    GRAD=4
  elif [ "${BATCH}" = "8" ]
  then
    GRAD=8
  elif [ "${BATCH}" = "4" ]
  then
    GRAD=16
  elif [ "${BATCH}" = "2" ]
  then
    GRAD=32
  elif [ "${BATCH}" = "1" ]
  then
    GRAD=64
  else
    echo "Unknown batch size ${BATCH}!"
    exit 125
  fi

  for LA in "ja" "es" "ko" "it" "de" "ru" "fr" "zh"
  do
    lmqg-train-search -c "lmqg_output/${MODEL_NAME}_${LA}quad" -d lmqg/qg_${LA}quad -m "${MODEL_ALIAS}" -b ${BATCH} -g ${GRAD} --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
    evaluate "${MODEL_NAME}_${LA}quad" ${LA} "qg_${LA}quad" "${MODEL_NAME//_/-}-${LA}quad-qg"
  done
}

# SQUAD with multilingual LM (for zeroshot-transfer)
lmqg-train-search -c lmqg_output/mt5_small_squad -d lmqg/qg_squad -m google/mt5-small -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language en --n-max-config 1
evaluate mt5_small_squad en qg_squad mt5-small-squad-qg
lmqg-train-search -c lmqg_output/mt5_base_squad -d lmqg/qg_squad -m google/mt5-base -b 32 -g 2 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language en --n-max-config 1
evaluate mt5_base_squad en qg_squad mt5-base-squad-qg
lmqg-train-search -c lmqg_output/mbart_large_cc25_squad -d lmqg/qg_squad -m facebook/mbart-large-cc25 -b 32 -g 2 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language en --n-max-config 1
evaluate mbart_large_cc25_squad en qg_squad mbart-large-cc25-squad-qg

# QG
mlqg "mt5_small" "google/mt5-small" "64"
mlqg "mt5_base" "google/mt5-base" "64"
mlqg "mbart_large_cc25" "facebook/mbart-large-cc25" "4"

###############
# SQUADSHIFTS #
###############
finetuning_ss () {
  for DATA_TYPE in "new_wiki" "nyt" "reddit" "amazon"
  do
    CKPT_FILE="lmqg_output/${2}-squadshifts-${DATA_TYPE}"
    lmqg-train-search -b "${3}" -g "${4}" "${5}" -d "lmqg/qg_squadshifts" --dataset-name "${DATA_TYPE}" -m "${1}" -c "${CKPT_FILE}" --n-max-config 1 --epoch-partial 1 -e 4
    lmqg-eval -m "${CKPT_FILE}/best_model" -e "${CKPT_FILE}/best_model/eval" --language "en" -d "lmqg/qg_squadshifts" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level sentence --dataset-name "${DATA_TYPE}"
    lmqg-push-to-hf -m "${CKPT_FILE}/best_model" -a "${6}-${DATA_TYPE}" -o "${7}-qg"
    rm -rf "${6}-${DATA_TYPE}"
  done
  }

finetuning_ss "lmqg/t5-large-squad" "qg_squadshifts/t5-large" "16" "4" "8" "t5-large-squadsquadshifts" "lmqg"
finetuning_ss "lmqg/t5-base-squad" "qg_squadshifts/t5-base" "8" "8" "16" "t5-base-squadshifts" "lmqg"
finetuning_ss "lmqg/t5-small-squad" "qg_squadshifts/t5-small" "32" "2" "4" "t5-small-squadshifts" "lmqg"
finetuning_ss "lmqg/bart-large-squad" "qg_squadshifts/bart-large" "32" "2" "4" "bart-large-squadshifts" "lmqg"
finetuning_ss "lmqg/bart-base-squad" "qg_squadshifts/bart-base" "8" "8" "16" "bart-base-squadshifts" "lmqg"


finetuning_ss "t5-large" "qg_squadshifts_vanilla/t5-large" "16" "4" "8" "t5-large-squadshifts-vanilla" "research-backup"
finetuning_ss "t5-base" "qg_squadshifts_vanilla/t5-base" "8" "8" "16" "t5-base-squadshifts-vanilla" "research-backup"
finetuning_ss "t5-small" "qg_squadshifts_vanilla/t5-small" "32" "2" "4" "t5-small-squadshifts-vanilla" "research-backup"
finetuning_ss "facebook/bart-large" "qg_squadshifts_vanilla/bart-large" "32" "2" "4" "bart-large-squadshifts-vanilla" "research-backup"
finetuning_ss "facebook/bart-base" "qg_squadshifts_vanilla/bart-base" "8" "8" "16" "bart-base-squadshifts-vanilla" "research-backup"

##########
# SUBJQA #
##########
finetuning_subjqa () {
  for DATA_TYPE in "books" "electronics" "grocery" "movies" "restaurants" "tripadvisor"
  do
    CKPT_FILE="lmqg_output/${2}_subjqa_${DATA_TYPE}"
    lmqg-train-search -b "${3}" -g "${4}" "${5}" -d "lmqg/qg_subjqa" --dataset-name "${DATA_TYPE}" -m "${1}" -p qg -c "${CKPT_FILE}" --n-max-config 1 --epoch-partial 1 -e 4
    lmqg-eval -m "${CKPT_FILE}/best_model" -e "${CKPT_FILE}/best_model/eval" --language "en" -d "lmqg/qg_subjqa" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level sentence --dataset-name "${DATA_TYPE}"
    lmqg-push-to-hf -m "${CKPT_FILE}/best_model" -a "${6}-${DATA_TYPE}" -o "${7}-qg"
    rm -rf "${6}-${DATA_TYPE}"
  done
  }

finetuning_subjqa "lmqg/t5-large-squad" "qg_subjqa/t5_large" "16" "4" "8" "t5-large-subjqa" "lmqg"
finetuning_subjqa "lmqg/t5-base-squad" "qg_subjqa/t5_base" "16" "4" "8" "t5-base-subjqa" "lmqg"
finetuning_subjqa "lmqg/t5-small-squad" "qg_subjqa/t5_small" "32" "2" "4" "t5-small-subjqa" "lmqg"
finetuning_subjqa "lmqg/bart-large-squad" "qg_subjqa/bart_large" "8" "8" "16" "bart-large-subjqa" "lmqg"
finetuning_subjqa "lmqg/bart-base-squad" "qg_subjqa/bart_base" "32" "2" "4" "bart-base-subjqa" "lmqg"

finetuning_subjqa "t5-large" "qg_subjqa_vanilla/t5_large" "16" "4" "8" "t5-large-subjqa-vanilla" "research-backup"
finetuning_subjqa "t5-base" "qg_subjqa_vanilla/t5_base" "16" "4" "8" "t5-base-subjqa-vanilla" "research-backup"
finetuning_subjqa "t5-small" "qg_subjqa_vanilla/t5_small" "32" "2" "4" "t5-small-subjqa-vanilla" "research-backup"
finetuning_subjqa "facebook/bart-large" "qg_subjqa_vanilla/bart_large" "8" "8" "16" "bart-large-subjqa-vanilla" "research-backup"
finetuning_subjqa "facebook/bart-base" "qg_subjqa_vanilla/bart_base" "8" "8" "16" "bart-base-subjqa-vanilla" "research-backup"


##################
# ABLATION STUDY #
##################
for MODEL in "t5-small" "t5-base" "t5-large" "bart-base" "bart-large"
do

  # Finetuning without paragraph
  lmqg-train-search --max-length 128 --max-length-eval 128 -m "${MODEL}" -p qg -i 'sentence_answer' -o 'question' -b 2 -g 32 64 128 256 -c "lmqg_output/${MODEL}-squad-qg-no-paragraph"
  lmqg-eval -m "lmqg_output/${MODEL}-squad-qg-no-paragraph/best_model" -e "lmqg_output/${MODEL}-squad-qg-no-paragraph/best_model/eval" -d "lmqg/qg_squad" -i "sentence_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/${MODEL}-squad-qg-no-paragraph/best_model" -a "${MODEL}-squad-qg-no-paragraph" -o "research-backup"

  # Finetuning without answer
  lmqg-train-search --max-length 128 --max-length-eval 128 -m "${MODEL}" -p qg -i 'paragraph_sentence' -o 'question' -b 2 -g 32 64 128 256 -c "lmqg_output/${MODEL}-squad-qg-no-answer"
  lmqg-eval -m "lmqg_output/${MODEL}-squad-qg-no-answer/best_model" -e "lmqg_output/${MODEL}-squad-qg-no-answer/best_model/eval" -d "lmqg/qg_squad" -i "paragraph_sentence" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/${MODEL}-squad-qg-no-answer/best_model" -a "${MODEL}-squad-qg-no-answer" -o "research-backup"

  # Finetuning without parameter optimization
  lmqg-train --epoch 10 -l 1.25e-5 --label-smoothing 0.1 -b 32 -g 1 -m "${MODEL}" -p qg -c "lmqg_output/${MODEL}-squad-qg-default"
  lmqg-eval -m "lmqg_output/${MODEL}-squad-qg-default/best_model" -e "lmqg_output/${MODEL}-squad-qg-default/best_model/eval" -d "lmqg/qg_squad" -i "paragraph_sentence" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/${MODEL}-squad-qg-default/best_model" -a "${MODEL}-squad-qg-default" -o "research-backup"

done