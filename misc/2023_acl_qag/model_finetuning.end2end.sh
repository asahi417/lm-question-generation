############
# ML SQuAD #
############
LA='zh'
LA='fr'
LA='ja'
LA='es'
LA='ko'
LA='de'
LA='it'
LA='ru'
MODEL="google/mt5-small"
MODEL_SHORT='mt5-small'
lmqg-train-search --use-auth-token -d "lmqg/qag_${LA}quad" -m "${MODEL}" -b 8 -g 8 16 -c "lmqg_output/${MODEL_SHORT}-${LA}quad-qag" -i 'paragraph' -o 'questions_answers' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256 --lr 1e-04 5e-04 1e-03
lmqg-eval -m "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model" -e "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model/eval" --language "${LA}" -d "lmqg/qag_${LA}quad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model" -a "${MODEL_SHORT}-${LA}quad-qag" -o "lmqg"
MODEL="google/mt5-base"
MODEL_SHORT='mt5-base'
lmqg-train-search --use-auth-token -d "lmqg/qag_${LA}quad" -m "${MODEL}" -b 2 -g 32 64 -c "lmqg_output/${MODEL_SHORT}-${LA}quad-qag" -i 'paragraph' -o 'questions_answers' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256 --lr 1e-04 5e-04 1e-03
lmqg-eval -m "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model" -e "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model/eval" --language "${LA}" -d "lmqg/qag_${LA}quad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model" -a "${MODEL_SHORT}-${LA}quad-qag" -o "lmqg"
MODEL="facebook/mbart-large-cc25"
MODEL_SHORT='mbart-large-cc25'
lmqg-train-search --use-auth-token -d "lmqg/qag_${LA}quad" -m "${MODEL}" -b 8 -g 8 16 -c "lmqg_output/${MODEL_SHORT}-${LA}quad-qag" -i 'paragraph' -o 'questions_answers' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256 --lr 1e-04 5e-04 1e-03
lmqg-eval -m "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model" -e "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model/eval" --language "${LA}" -d "lmqg/qag_${LA}quad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/${MODEL_SHORT}-${LA}quad-qag/best_model" -a "${MODEL_SHORT}-${LA}quad-qag" -o "lmqg"

#########
# SQuAD #
#########
lmqg-train-search -d "lmqg/qag_squad" -m "t5-small" -b 32 -g 2 4 -c "lmqg_output/t5-small-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/t5-small-squad-qag/best_model" -e "lmqg_output/t5-small-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-small-squad-qag/best_model" -a "t5-small-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "t5-base"  -b 8 -g 8 16 -c "lmqg_output/t5-base-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/t5-base-squad-qag/best_model" -e "lmqg_output/t5-base-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-base-squad-qag/best_model" -a "t5-base-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5-large-squad-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --n-max-config 2 --epoch-partial 5 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/t5-large-squad-qag/best_model" -e "lmqg_output/t5-large-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-large-squad-qag/best_model" -a "t5-large-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "facebook/bart-base" -b 16 -g 4 8 -c "lmqg_output/bart-base-squad-qag" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/bart-base-squad-qag/best_model" -e "lmqg_output/bart-base-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart-base-squad-qag/best_model" -a "bart-base-squad-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_squad" -m "facebook/bart-large" -b 8 -g 8 16 -c "lmqg_output/bart-large-squad-qag" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 256 --max-length-output 256
lmqg-eval -m "lmqg_output/bart-large-squad-qag/best_model" -e "lmqg_output/bart-large-squad-qag/best_model/eval" --language "en" -d "lmqg/qag_squad" -i "paragraph" -o 'questions_answers' --max-length-output 256 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart-large-squad-qag/best_model" -a "bart-large-squad-qag" -o "lmqg"

###########
# TweetQA #
###########
lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5-small-tweetqa-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5-small-tweetqa-qag/best_model" -e "lmqg_output/t5-small-tweetqa-qag/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-small-tweetqa-qag/best_model" -a "t5-small-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-base"  -b 32 -g 2 4 -c "lmqg_output/t5-base-tweetqa-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5-base-tweetqa-qag/best_model" -e "lmqg_output/t5-base-tweetqa-qag/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-base-tweetqa-qag/best_model" -a "t5-base-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5-large-tweetqa-qag" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5-large-tweetqa-qag/best_model" -e "lmqg_output/t5-large-tweetqa-qag/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-large-tweetqa-qag/best_model" -a "t5-large-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "facebook/bart-base" -b 32 -g 2 4 -c "lmqg_output/bart-base-tweetqa-qag" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/bart-base-tweetqa-qag/best_model" -e "lmqg_output/bart-base-tweetqa-qag/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart-base-tweetqa-qag/best_model" -a "bart-base-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "facebook/bart-large" -b 32 -g 2 4 -c "lmqg_output/bart-large-tweetqa-qag" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/bart-large-tweetqa-qag/best_model" -e "lmqg_output/bart-large-tweetqa-qag/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart-large-tweetqa-qag/best_model" -a "bart-large-tweetqa-qag" -o "lmqg"

############
# Ablation #
############
# Without prefix
lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5-small-tweetqa-qag-np" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5-small-tweetqa-qag-np/best_model" -e "lmqg_output/t5-small-tweetqa-qag-np/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-small-tweetqa-qag-np/best_model" -a "t5-small-tweetqa-qag-np" -o "research-backup"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-base"  -b 32 -g 2 4 -c "lmqg_output/t5-base-tweetqa-qag-np" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5-base-tweetqa-qag-np/best_model" -e "lmqg_output/t5-base-tweetqa-qag-np/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-base-tweetqa-qag-np/best_model" -a "t5-base-tweetqa-qag-np" -o "research-backup"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5-large-tweetqa-qag-np" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5-large-tweetqa-qag-np/best_model" -e "lmqg_output/t5-large-tweetqa-qag-np/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5-large-tweetqa-qag-np/best_model" -a "t5-large-tweetqa-qag-np" -o "research-backup"
