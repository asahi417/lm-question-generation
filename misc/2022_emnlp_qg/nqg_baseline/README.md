# Non-LM QG Baseline
Here we explain how we collect predictions for our analysis from ***non*** language model (non-LM) approach.

## Get Raw Model Prediction
As our non-LM baseline, we employ [Neural Question Generation](https://arxiv.org/abs/1705.00106) (NQG), which is an LSTM based seq2seq model for the question generation task.
To get the prediction from the NQG model, we follow the author's [github repo](https://github.com/xinyadu/nqg) with a trained model checkpoint 
shared [here](https://github.com/xinyadu/nqg/issues/9).
The prediction file can be found at [here](nonlm.sample.test.hyp.txt), and following steps allow to reproduce the prediction file from scratch.

***Step 1:*** Install the repo and create a directory to put the model checkpoint. Make sure that your environment has `torch` installed.
```shell
git clone https://github.com/xinyadu/nqg
cd nqg/sentence
mkdir model
wget https://github.com/asahi417/lm-question-generation/releases/download/0.0.0/squad-test-sentence.txt
```

***Step 2:*** Download the model checkpoint from [here](https://drive.google.com/file/d/16Gi1oZr3mEGMEUCsOQ3whFfDlv8IAyzG/view?usp=sharing), and 
locate it under `model` directory.

***Step 3:*** Overwrite config file `config-trans` as below to specify the file to generate prediction on. If you use GPUs, set
`gpuid=1` instead.

```shell
src=./squad-test-sentence.txt 
beam_size=3
gpuid=0
replace_unk=true
```

***Step 3:*** Get prediction by following command. 
```shell
th translate.lua -model model/model.t7 -config config-trans
```

Once the process was finished, you should get the prediction file `pred.txt`. 
To use the file on the next step, rename the file as `nonlm.sample.test.hyp.txt`.

## Handling Preprocessing Artifacts
NQG relies on preprocessing including lower-casing and delimiter splitting,
so the generated questions inherit such artifacts as shown below examples.

```
where was saint denis reputedly located ?
what percentage of dna is no less than 20 % ?
what did the term `` scientific socialism '' refer to ?
```

Here we aime to restore a natural text from the raw prediction of the NQG model as below.

```
Where was Saint Denis reputedly located?
What percentage of DNA is no less than 20%?
What did the term "Scientific Socialism" refer to?
```

## Evaluate Prediction to Get Comparable Metrics
Compute metrics (BLEU4/METEOR/ROUGE) with the SQuAD Reference files.

```shell
lmqg-eval --hyp-test nonlm_fixed.sample.test.hyp.txt -e ./
lmqg-eval --hyp-test nonlm_fixed.sample.test.hyp.txt -e ./ --prediction-level answer
```

The output includes [`metric.first.answer.json`](metric.first.answer.json) (answer-level metric),
and [`metric.first.sentence.json`](metric.first.sentence.json) (sentence-level metric),
