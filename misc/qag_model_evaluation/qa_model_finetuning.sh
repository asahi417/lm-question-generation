
LM="distilbert-base-uncased"
for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
do
  lmqg-qae -m "${LM}" -d "lmqg/qa_squadshifts" -n "${NAME}" --output-dir "qa_eval_output/gold_qa/${LM}.qa_squadshifts.${NAME}"
done