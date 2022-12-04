# Out-of-domain Evaluation of QG models

# SQUAD --> SQUADSHIFTS
for QG_MODEL in "t5-small" "t5-base" "t5-large" "bart-base" "bart-large"
do
  git clone "https://huggingface.co/lmqg/${QG_MODEL}-squad-qg"
  for DATA_TYPE in "new_wiki" "nyt" "reddit" "amazon"
  do
    lmqg-eval --batch-size 4 -m "${QG_MODEL}-squad-qg" -e "${QG_MODEL}-squad-qg/eval_ood" -d "lmqg/qg_squadshifts" --dataset-name "${DATA_TYPE}" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  done
  lmqg-push-to-hf -m "${QG_MODEL}-squad-qg" -a "${QG_MODEL}-squad-qg" -o "lmqg"
done

# SQUAD --> SUBJQA
for QG_MODEL in "t5-small" "t5-base" "t5-large" "bart-base" "bart-large"
do
  git clone "https://huggingface.co/lmqg/${QG_MODEL}-squad-qg"
  for DATA_TYPE in "books" "electronics" "grocery" "movies" "restaurants" "tripadvisor"
  do
    lmqg-eval --batch-size 4 -m "${QG_MODEL}-squad-qg" -e "${QG_MODEL}-squad-qg/eval_ood" -d "lmqg/qg_subjqa" --dataset-name "${DATA_TYPE}" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  done
  lmqg-push-to-hf -m "${QG_MODEL}-squad-qg" -a "${QG_MODEL}-squad-qg" -o "lmqg"
done

# Multilingual LM trained on SQUAD --> Multilingual QG (non-English)
for QG_MODEL in "mt5-small" "mt5-base" "mbart-large-cc25"
do
  git clone "https://huggingface.co/lmqg/${QG_MODEL}-squad-qg"
  for LA in "ja" "es" "de" "ru" "ko" "fr" "it"
  do
    lmqg-eval --batch-size 4 -m "${QG_MODEL}-squad-qg" -e "${QG_MODEL}-squad-qg/eval_ood" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  done
  lmqg-push-to-hf -m "${QG_MODEL}-squad-qg" -a "${QG_MODEL}-squad-qg" -o "lmqg"
done
