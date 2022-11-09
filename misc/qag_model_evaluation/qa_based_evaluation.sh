
LM="distilbert-base-uncased"

# QA metric on the gold qa pairs
for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
do
  lmqg-qae -m "${LM}" -d 'lmqg/qa_squadshifts' -n "${NAME}" --output-dir "qae_result/gold_qa/${LM}.${NAME}"
done