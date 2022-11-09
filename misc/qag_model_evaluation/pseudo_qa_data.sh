
BATCH=14
# Generate Pseudo QA Dataset on SQuADShifts
qa_generation_squadshifts () {
  MODEL=${1}
#  NAME='reddit'
#  lmqg-generate-qa -a -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" \
#  -e "qa_dataset/qa_squadshifts.${MODEL}.${NAME}"
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa -a -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" \
    -e "qa_dataset/qa_squadshifts.${MODEL}.${NAME}"
  done
}

# Generate Pseudo Question Dataset on SQuADShifts (answer is taken from the gold reference)
q_generation_squadshifts () {
  MODEL=${1}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" \
    -e "qa_dataset/qa_squadshifts.${MODEL}.${NAME}"
  done
}


qa_generation_squadshifts "t5-large-squad-multitask"
q_generation_squadshifts "t5-large-squad"
qa_generation_squadshifts "t5-base-squad-multitask"
q_generation_squadshifts "bart-large-squad"
qa_generation_squadshifts "t5-small-squad-multitask"
q_generation_squadshifts "t5-base-squad"
q_generation_squadshifts "t5-small-squad"
q_generation_squadshifts "bart-base-squad"

#d = load_dataset('json', data_files='./qa_dataset/qa_squadshifts.bart-base-squad.amazon/test.jsonl')

