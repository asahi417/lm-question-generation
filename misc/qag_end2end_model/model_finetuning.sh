# With prefix
lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5_small_tweetqa" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_small_tweetqa/best_model" -e "lmqg_output/t5_small_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_small_tweetqa/best_model" -a "t5-small-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-base"  -b 32 -g 2 4 -c "lmqg_output/t5_base_tweetqa" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_base_tweetqa/best_model" -e "lmqg_output/t5_base_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_base_tweetqa/best_model" -a "t5-base-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5_large_tweetqa" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_large_tweetqa/best_model" -e "lmqg_output/t5_large_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_large_tweetqa/best_model" -a "t5-large-tweetqa-qag" -o "lmqg"

# Without prefix
lmqg-train-search -d "lmqg/qag_tweetqa" -m "facebook/bart-base" -b 32 -g 4 8 -c "lmqg_output/bart_base_tweetqa" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/bart_base_tweetqa/best_model" -e "lmqg_output/bart_base_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart_base_tweetqa/best_model" -a "bart-base-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5_small_tweetqa_np" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_small_tweetqa_np/best_model" -e "lmqg_output/t5_small_tweetqa_np/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_small_tweetqa_np/best_model" -a "t5-small-tweetqa-qag-np" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-base"  -b 32 -g 2 4 -c "lmqg_output/t5_base_tweetqa_np" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_base_tweetqa_np/best_model" -e "lmqg_output/t5_base_tweetqa_np/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_base_tweetqa_np/best_model" -a "t5-base-tweetqa-qag-np" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5_large_tweetqa_np" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_large_tweetqa_np/best_model" -e "lmqg_output/t5_large_tweetqa_np/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_large_tweetqa_np/best_model" -a "t5-large-tweetqa-qag-np" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "facebook/bart-large" -b 32 -g 4 8 -c "lmqg_output/bart_large_tweetqa" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/bart_large_tweetqa/best_model" -e "lmqg_output/bart_large_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "answer" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart_large_tweetqa/best_model" -a "bart-large-tweetqa-qag" -o "lmqg"

# extra evaluation
#for LM in "bart-base-tweetqa-qag" "bart-large-tweetqa-qag" "t5-small-tweetqa-qag" "t5-base-tweetqa-qag" "t5-large-tweetqa-qag" "t5-small-tweetqa-qag-np" "t5-base-tweetqa-qag-np" "t5-large-tweetqa-qag-np"
for LM in "bart-large-tweetqa-qag" "t5-small-tweetqa-qag-np" "t5-base-tweetqa-qag-np" "t5-large-tweetqa-qag-np"
do
  git clone "https://huggingface.co/lmqg/${LM}"
  rm -rf "${LM}/eval/metric.first.answer.paragraph.questions_answers.lmqg_qag_tweetqa.default.json"
  lmqg-eval -m "${LM}" -e "${LM}/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-level "answer" --max-length-output 128 --max-length 256
  lmqg-push-to-hf -m "${LM}" -a "${LM}" -o "lmqg"
  rm -rf "${LM}"
done
