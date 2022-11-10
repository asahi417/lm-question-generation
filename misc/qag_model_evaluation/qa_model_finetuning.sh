
LM="distilbert-base-uncased"

# QAE with Gold QA dataset
for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
do
  lmqg-qae -m "${LM}" -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/gold_qa/${LM}.qa_squadshifts.${NAME}"
done

# QAE with Generated Pseudo QA dataset
QAE () {
  MODEL=${1}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-qae -m "${LM}" -d "json" \
    --dataset-train "qa_dataset/${MODEL}.${NAME}/train.jsonl" \
    --dataset-validation "qa_dataset/${MODEL}.${NAME}/validation.jsonl" \
    --dataset-test "qa_dataset/${MODEL}.${NAME}/test.jsonl" \
    --output-dir "qa_eval_output/silver_qa.${MODEL}/${LM}.qa_squadshifts.${NAME}"
  done
}


QAE "t5-large-squad"
QAE "t5-base-squad"
QAE "t5-small-squad"
QAE "bart-large-squad"
QAE "bart-base-squad"
