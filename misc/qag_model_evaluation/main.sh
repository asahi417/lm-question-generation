
# Generate Pseudo QA Dataset on SQuADShifts
qa_generation_squadshifts () {
  MODEL=${1}
  BATCH=${2}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa -a -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" \
    -e "qa_dataset/qa_squadshifts.${MODEL}.${NAME}"
  done
}

# Generate Pseudo Question Dataset on SQuADShifts (answer is taken from the gold reference)
q_generation_squadshifts () {
  MODEL=${1}
  BATCH=${2}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" \
    -e "qa_dataset/qa_squadshifts.${MODEL}.${NAME}"
  done
}


qa_generation_squadshifts "t5-small-squad-multitask" 256
qa_generation_squadshifts "t5-base-squad-multitask" 256
qa_generation_squadshifts "t5-large-squad-multitask" 256

q_generation_squadshifts "t5-small-squad" 256
q_generation_squadshifts "t5-base-squad" 256
q_generation_squadshifts "t5-large-squad" 256
q_generation_squadshifts "bart-base-squad" 256
q_generation_squadshifts "bart-large-squad" 256

