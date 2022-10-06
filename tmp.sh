# TODO ADD MULTITASK MODELS

for MODEL in "bart-base" "bart-large" "t5-small" "t5-base" "t5-large"
do
  MODEL_PATH="${MODEL}-squad"
  git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
  lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
  rm -rf ${MODEL_PATH}
done

for MODEL in "t5-small" "t5-base"
do
  MODEL_PATH="${MODEL}-squad-multitask"
  git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
  lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
  rm -rf ${MODEL_PATH}
done

for MODEL in "mt5-small" "mt5-base" "mbart-large-cc25"
do
  for DATA in "squad" "esquad" "frquad" "koquad" "ruquad" "dequad" "itquad" "jaquad"
  do
    MODEL_PATH="${MODEL}-${DATA}"
    git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
    lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
    rm -rf ${MODEL_PATH}
  done
done

MODEL="mt5-small"
for DATA in "esquad" "frquad" "koquad" "ruquad" "dequad" "itquad" "jaquad"
do
  MODEL_PATH="${MODEL}-${DATA}-multitask"
  git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
  lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
  rm -rf ${MODEL_PATH}
done


for MODEL in "bart-base" "bart-large" "t5-small" "t5-base" "t5-large"
do
  for VERSION in "default" "no-paragraph" "no-answer"
  do
    MODEL_PATH="${MODEL}-squad-${VERSION}"
    git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
    lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
    rm -rf ${MODEL_PATH}
  done
done

for MODEL in "bart-base" "bart-large" "t5-small" "t5-base" "t5-large"
do
  DATA="subjqa"
  for TYPE in "books" "electronics" "movies" "grocery" "restaurants" "tripadvisor"
  do
    for VERSION in "" "-vanilla"
    do
      MODEL_PATH="${MODEL}-${DATA}${VERSION}-${TYPE}"
      git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
      lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
      rm -rf ${MODEL_PATH}
    done
  done
done

for MODEL in "bart-base" "bart-large" "t5-small" "t5-base" "t5-large"
do
  DATA="squadshifts"
  for TYPE in "amazon" "new_wiki" "nyt" "reddit"
  do
    for VERSION in "" "-vanilla"
    do
      MODEL_PATH="${MODEL}-${DATA}${VERSION}-${TYPE}"
      git clone "https://huggingface.co/lmqg/${MODEL_PATH}"
      lmqg-push-to-hf -m ${MODEL_PATH} -a ${MODEL_PATH} -o lmqg --skip-model-upload
      rm -rf ${MODEL_PATH}
    done
  done
done


