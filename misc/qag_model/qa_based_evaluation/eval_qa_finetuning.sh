# QAEval on Gold QA dataset
for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
do
  lmqg-qae -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/gold_qa/qa_squadshifts.${NAME}" --down-sample-size-train 1000 --down-sample-size-valid 500
done


# QAE with Generated Pseudo QA dataset
QAE () {
  ANCHOR_MODEL=${1}
  QAG_TYPE=${2}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-qae -d "lmqg/qa_squadshifts_synthetic" -n "${ANCHOR_MODEL}.${QAG_TYPE}.${NAME}" --output-dir "qa_eval_output/silver_qa.${ANCHOR_MODEL}.${QAG_TYPE}/qa_squadshifts.${NAME}" --down-sample-size-train 1000 --down-sample-size-valid 500
  done
}

QAE_LOCAL () {
  # Same function as `QAE` but working with local files
  ANCHOR_MODEL=${1}
  QAG_TYPE=${2}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-qae -d "json" \
      --dataset-train "qa_squadshifts_synthetic/${ANCHOR_MODEL}.${QAG_TYPE}.${NAME}/train.jsonl" \
      --dataset-validation "qa_squadshifts_synthetic/${ANCHOR_MODEL}.${QAG_TYPE}.${NAME}/validation.jsonl" \
      --output-dir "qa_eval_output/silver_qa.${ANCHOR_MODEL}.${QAG_TYPE}/qa_squadshifts.${NAME}" \
      --down-sample-size-train 1000 --down-sample-size-valid 500
    lmqg-qae --overwrite --skip-training -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/silver_qa.${ANCHOR_MODEL}.${QAG_TYPE}/qa_squadshifts.${NAME}"
  done
}

QAE "t5-small-squad" "qg_reference"
QAE "t5-base-squad" "qg_reference"
QAE "t5-large-squad" "qg_reference"
QAE "bart-base-squad" "qg_reference"
QAE "bart-large-squad" "qg_reference"

QAE "t5-small-squad" "multitask"
QAE "t5-base-squad" "multitask"
QAE "t5-large-squad" "multitask"
QAE "bart-base-squad" "multitask"
QAE "bart-large-squad" "multitask"

QAE "t5-small-squad" "pipeline"
QAE "t5-base-squad" "pipeline"
QAE "t5-large-squad" "pipeline"
QAE "bart-base-squad" "pipeline"
QAE "bart-large-squad" "pipeline"

QAE "t5-small-squad" "end2end"
QAE "t5-base-squad" "end2end"
QAE "t5-large-squad" "end2end"
QAE "bart-base-squad" "end2end"
QAE "bart-large-squad" "end2end"


