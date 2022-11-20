

LM="distilbert-base-uncased"
# QAE with Gold QA dataset
for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
do
  lmqg-qae -m "${LM}" -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/gold_qa/${LM}.qa_squadshifts.${NAME}"
done

# QAE with Generated Pseudo QA dataset
QAE () {
  LM="distilbert-base-uncased"
  MODEL=${1}
  EVAL_STEP=${2}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-qae -m "${LM}" -d "lmqg/qa_squadshifts_pseudo" -n "${NAME}.${MODEL}" --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}" --eval-step "${EVAL_STEP}"
#    lmqg-qae -m "${LM}" -d "json" \
#      --dataset-train "qa_squadshifts_pseudo/${MODEL}.${NAME}/train.jsonl" \
#      --dataset-validation "qa_squadshifts_pseudo/${MODEL}.${NAME}/validation.jsonl" \
#      --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}"
#    lmqg-qae --overwrite --skip-training -m "${LM}" -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}"
  done
}

QAE "t5-large-squad" 50
QAE "t5-base-squad" 50
QAE "t5-small-squad" 50
QAE "bart-base-squad" 50
QAE "bart-large-squad" 50
QAE "t5-large-squad-multitask-qg" 50
QAE "t5-base-squad-multitask-qg" 50
QAE "t5-small-squad-multitask-qg" 50

QAE "t5-large-squad-multitask" 250
QAE "t5-base-squad-multitask" 250
QAE "t5-small-squad-multitask" 250


#QAE "t5-large-squad-multitask" 500
#QAE "t5-base-squad-multitask" 500
#QAE "t5-small-squad-multitask" 500



QAE_FILTERED () {
  LM="distilbert-base-uncased"
  MODEL=${1}
  EVAL_STEP=${2}
  FILTERING=${3}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-qae -m "${LM}" -d "lmqg/qa_squadshifts_pseudo" -n "${NAME}.${MODEL}.${FILTERING}" --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}.${FILTERING}" --eval-step "${EVAL_STEP}"
  done
#  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
#  do
#    lmqg-qae -m "${LM}" -d "json" \
#      --dataset-train "qa_squadshifts_pseudo/${MODEL}.${NAME}.${FILTERING}/train.jsonl" \
#      --dataset-validation "qa_squadshifts_pseudo/${MODEL}.${NAME}.${FILTERING}/validation.jsonl" \
#      --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}.${FILTERING}"
#    lmqg-qae --overwrite --skip-training -m "${LM}" -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}.${FILTERING}"
#  done
}


QAE_FILTERED "t5-large-squad-multitask" 50 "perplexity_answer"
QAE_FILTERED "t5-large-squad-multitask" 50 "perplexity_question"

QAE_FILTERED "t5-base-squad-multitask" 50 "perplexity_answer"
QAE_FILTERED "t5-base-squad-multitask" 50 "perplexity_question"
QAE_FILTERED "t5-small-squad-multitask" 50 "perplexity_answer"
QAE_FILTERED "t5-small-squad-multitask" 50 "perplexity_question"

