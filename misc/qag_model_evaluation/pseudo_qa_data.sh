BATCH=16

# Generate Pseudo QA Dataset on SQuADShifts
qa_generation_squadshifts () {
  MODEL=${1}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    # QA generation
    lmqg-generate-qa --overwrite -a -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${MODEL}.${NAME}"
    # Question generation on the gold answer
#    lmqg-generate-qa -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${MODEL}-qg.${NAME}"
  done
}

qa_generation_squadshifts "t5-large-squad-multitask"
qa_generation_squadshifts "t5-base-squad-multitask"
qa_generation_squadshifts "t5-small-squad-multitask"


# Generate Pseudo Question Dataset on SQuADShifts (answer is taken from the gold reference)
q_generation_squadshifts () {
  MODEL=${1}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${MODEL}.${NAME}"
  done
}

q_generation_squadshifts "t5-large-squad"
q_generation_squadshifts "t5-base-squad"
q_generation_squadshifts "t5-small-squad"
q_generation_squadshifts "bart-large-squad"
q_generation_squadshifts "bart-base-squad"
