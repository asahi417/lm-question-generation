
lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-small" -b 64 -g 1 2 -c "lmqg_output/t5_small_tweetqa" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_small_tweetqa/best_model" -e "lmqg_output/t5_small_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "sentence" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_small_tweetqa/best_model" -a "t5-small-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-base"  -b 64 -g 1 2 -c "lmqg_output/t5_base_tweetqa" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_base_tweetqa/best_model" -e "lmqg_output/t5_base_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "sentence" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_base_tweetqa/best_model" -a "t5-base-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5_large_tweetqa" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_large_tweetqa/best_model" -e "lmqg_output/t5_large_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "sentence" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_large_tweetqa/best_model" -a "t5-large-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "t5-large" -b 16 -g 4 8 -c "lmqg_output/t5_large_tweetqa" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/t5_large_tweetqa/best_model" -e "lmqg_output/t5_large_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "sentence" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/t5_large_tweetqa/best_model" -a "t5-large-tweetqa-qag-np" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "facebook/bart-base" -b 32 -g 4 8 -c "lmqg_output/bart_base_tweetqa" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/bart_base_tweetqa/best_model" -e "lmqg_output/bart_base_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "sentence" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart_base_tweetqa/best_model" -a "bart-base-tweetqa-qag" -o "lmqg"

lmqg-train-search -d "lmqg/qag_tweetqa" -m "facebook/bart-large" -b 32 -g 4 8 -c "lmqg_output/bart_large_tweetqa" -i 'paragraph' -o 'questions_answers' --epoch-partial 10 -e 15 --max-length-output-eval 128 --max-length-output 128 --max-length-eval 256 --max-length 256
lmqg-eval -m "lmqg_output/bart_large_tweetqa/best_model" -e "lmqg_output/bart_large_tweetqa/best_model/eval" --language "en" -d "lmqg/qag_tweetqa" -i "paragraph" -o 'questions_answers' --prediction-aggregation "first" --prediction-level "sentence" --max-length-output 128 --max-length 256
lmqg-push-to-hf -m "lmqg_output/bart_large_tweetqa/best_model" -a "bart-large-tweetqa-qag" -o "lmqg"

