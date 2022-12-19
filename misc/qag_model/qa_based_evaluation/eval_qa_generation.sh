#--use-reference-answer
# Generate Pseudo QA Dataset on SQuADShifts (Piepeline)
qa_generation_squadshifts_pipeline () {
  ANCHOR_MODEL=${1}
  BATCH=16
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa --overwrite -m "lmqg/${ANCHOR_MODEL}-qg" --model-ae "lmqg/${ANCHOR_MODEL}-ae" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${ANCHOR_MODEL}.pipeline.${NAME}" --compute-ppl
  done
}

qa_generation_squadshifts_pipeline "t5-small-squad"
qa_generation_squadshifts_pipeline "t5-base-squad"
qa_generation_squadshifts_pipeline "t5-large-squad"
qa_generation_squadshifts_pipeline "bart-base-squad"
qa_generation_squadshifts_pipeline "bart-large-squad"

# Generate Pseudo QA Dataset on SQuADShifts (E2E)
qa_generation_squadshifts_end2end () {
  ANCHOR_MODEL=${1}
  BATCH=16
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    lmqg-generate-qa --overwrite -a -m "lmqg/${ANCHOR_MODEL}-qag" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${ANCHOR_MODEL}.end2end.${NAME}" --compute-ppl
  done
}

qa_generation_squadshifts_end2end "t5-small-squad"
qa_generation_squadshifts_end2end "t5-base-squad"
qa_generation_squadshifts_end2end "t5-large-squad"
qa_generation_squadshifts_end2end "bart-base-squad"
qa_generation_squadshifts_end2end "bart-large-squad"


# Generate Pseudo QA Dataset on SQuADShifts
qa_generation_squadshifts () {
  MODEL=${1}
  for NAME in 'amazon' 'new_wiki' 'nyt' 'reddit'
  do
    # QA generation
    lmqg-generate-qa --overwrite -a -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${MODEL}.${NAME}" --compute-ppl
    lmqg-generate-qa --overwrite -a -m "lmqg/${MODEL}-qg" --model-ae "lmqg/${MODEL}-ae" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${MODEL}.${NAME}" --compute-ppl
#    # Question generation on the gold answer
#    lmqg-generate-qa -m "lmqg/${MODEL}" -l "en" -d "lmqg/qa_squadshifts" -n "${NAME}" -b "${BATCH}" -e "qa_squadshifts_pseudo/${MODEL}-qg.${NAME}"
  done
}

qa_generation_squadshifts "t5-large-squad-qg-ae"
qa_generation_squadshifts "t5-base-squad-qg-ae"
qa_generation_squadshifts "t5-small-squad-qg-ae"

qa_generation_squadshifts "t5-large-squad-qg-ae"
qa_generation_squadshifts "t5-base-squad-qg-ae"
qa_generation_squadshifts "t5-small-squad-qg-ae"


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
