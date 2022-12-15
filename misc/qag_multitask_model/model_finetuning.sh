# Multitask QAG Model
# train, eval QG, eval AE, eval AR (answer metric), eval QAG, push-to-hub

lmqg-train-search -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5-large-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-large-squad-qg-ae/best_model" -e "lmqg_output/t5-large-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-qg-ae/best_model" -a "t5-large-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "t5-base"  -b 32 -g 2 4 8 -c "lmqg_output/t5-base-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-base-squad-qg-ae/best_model" -e "lmqg_output/t5-base-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-qg-ae/best_model" -a "t5-base-squad-qg-ae" -o "lmqg"

lmqg-train-search -m "t5-small" -b 64 -g 1 2 4 -c "lmqg_output/t5-small-squad-qg-ae" -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_answer" -o 'question'
lmqg-eval -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag -m "lmqg_output/t5-small-squad-qg-ae/best_model" -e "lmqg_output/t5-small-squad-qg-ae/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-qg-ae/best_model" -a "t5-small-squad-qg-ae" -o "lmqg"

# Multitask QAG Model for non-English
mlqg_answer () {
  MODEL_NAME=${1}
  MODEL_ALIAS=${2}
  BATCH=${3}
  GRAD=${4}
  for LA in "ja" "es" "ko" "it" "de" "ru" "fr"
  do
    lmqg-train-search --use-auth-token -c "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae" -d "lmqg/qg_${LA}quad" -m "${MODEL_ALIAS}" -b ${BATCH} -g ${GRAD} --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" -o 'question'
    lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-eval-qa --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
    lmqg-eval-qag --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad"
    lmqg-push-to-hf -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -a "${MODEL_NAME}-${LA}quad-qg-ae" -o "lmqg"
  done
}

mlqg_answer "mt5-small" "google/mt5-small" "64" "1"
mlqg_answer "mt5-base" "google/mt5-base" "32" "2"

# TODO: Additional training for German
LA="de"
MODEL_ALIAS="google/mt5-base"
MODEL_NAME="mt5-base"
lmqg-train-search --use-auth-token -c "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae" -d "lmqg/qg_${LA}quad" -m "${MODEL_ALIAS}" -b 32 -g 4 8 --lr 1e-04 5e-04 --epoch-partial 5 -e 20 --label-smoothing 0.15 --language "${LA}" --n-max-config 1 -i 'paragraph_answer' 'paragraph_sentence' -o 'question' 'answer' -p 'qg' 'ae'
lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" -o 'question'
lmqg-eval --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qag --use-auth-token -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -e "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad"
lmqg-push-to-hf -m "lmqg_output/${MODEL_NAME}-${LA}quad-qg-ae/best_model" -a "${MODEL_NAME}-${LA}quad-qg-ae" -o "lmqg"

# Ablation: answer extraction only model
lmqg-train-search -m "t5-large" -b 4 -g 16 32 -c "lmqg_output/t5-large-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-large-squad-ae/best_model" -e "lmqg_output/t5-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-large-squad-ae/best_model" -e "lmqg_output/t5-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-ae/best_model" -a "t5-large-squad-ae" -o "lmqg"

lmqg-train-search -m "t5-base" -b 16 -g 4 8 -c "lmqg_output/t5-base-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-base-squad-ae/best_model" -e "lmqg_output/t5-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-base-squad-ae/best_model" -e "lmqg_output/t5-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-ae/best_model" -a "t5-base-squad-ae" -o "lmqg"

lmqg-train-search -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5-small-squad-ae" -i 'paragraph_sentence' -o 'answer' -p 'ae'
lmqg-eval -m "lmqg_output/t5-small-squad-ae/best_model" -e "lmqg_output/t5-small-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/t5-small-squad-ae/best_model" -e "lmqg_output/t5-small-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-ae/best_model" -a "t5-small-squad-ae" -o "lmqg"

lmqg-train-search -m "facebook/bart-large" -b 32 -g 2 4 -c "lmqg_output/bart-large-squad-ae" -i 'paragraph_sentence' -o 'answer'
lmqg-eval -m "lmqg_output/bart-large-squad-ae/best_model" -e "lmqg_output/bart-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-large-squad-ae/best_model" -e "lmqg_output/bart-large-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-large-squad-ae/best_model" -a "bart-large-squad-ae" -o "lmqg"

lmqg-train-search -m "facebook/bart-base" -b 16 -g 4 8 -c "lmqg_output/bart-base-squad-ae" -i 'paragraph_sentence' -o 'answer'
lmqg-eval -m "lmqg_output/bart-base-squad-ae/best_model" -e "lmqg_output/bart-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-eval-qa -m "lmqg_output/bart-base-squad-ae/best_model" -e "lmqg_output/bart-base-squad-ae/best_model/eval" --language "en" -d "lmqg/qg_squad" -i "paragraph_sentence" -o 'answer'
lmqg-push-to-hf -m "lmqg_output/bart-base-squad-ae/best_model" -a "bart-base-squad-ae" -o "lmqg"

# Evaluate pipeline QAG: QG + QA models
git clone "https://huggingface.co/lmqg/t5-small-squad-qg"
lmqg-eval-qag -m "t5-small-squad-qg" --model-ae "lmqg/t5-small-squad-ae" -e "t5-small-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
cd "t5-small-squad-qg" && git add . && git commit -m "eval pipeline" && git push && cd ..

git clone "https://huggingface.co/lmqg/t5-base-squad-qg"
lmqg-eval-qag -m "t5-base-squad-qg" --model-ae "lmqg/t5-base-squad-ae" -e "t5-base-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
cd "t5-base-squad-qg" && git add . && git commit -m "eval pipeline" && git push && cd ..

git clone "https://huggingface.co/lmqg/t5-large-squad-qg"
lmqg-eval-qag -m "t5-large-squad-qg" --model-ae "lmqg/t5-large-squad-ae" -e "t5-large-squad-qg/eval_pipeline" -d "lmqg/qg_squad" --language "en"
cd "t5-large-squad-qg" && git add . && git commit -m "eval pipeline" && git push && cd ..