
qag_metric () {
  MODEL=${1}
  DATA=${2}
  LA=${3}
  git clone "https://huggingface.co/lmqg/${MODEL}"
  lmqg-eval-qag -m "lmqg/${MODEL}" -e "${MODEL}/eval" -d "${DATA}" --language "${LA}" --batch-size 64
  lmqg-push-to-hf -m "${MODEL}" -a "${MODEL}" -o "lmqg"
#  rm -rf "${MODEL}"
}


# Multitask QAG models
qag_metric 't5-small-squad-multitask' 'lmqg/qg_squad' "en"
qag_metric 't5-base-squad-multitask' 'lmqg/qg_squad' "en"
qag_metric 't5-large-squad-multitask' 'lmqg/qg_squad' "en"
qag_metric 'mt5-small-dequad-multitask' 'lmqg/qg_dequad' "de"
qag_metric 'mt5-small-esquad-multitask' 'lmqg/qg_esquad' "es"
qag_metric 'mt5-small-itquad-multitask' 'lmqg/qg_itquad' "it"
qag_metric 'mt5-small-jaquad-multitask' 'lmqg/qg_jaquad' "ja"
qag_metric 'mt5-small-koquad-multitask' 'lmqg/qg_koquad' "ko"
qag_metric 'mt5-small-ruquad-multitask' 'lmqg/qg_ruquad' "ru"
qag_metric 'mt5-small-frquad-multitask' 'lmqg/qg_frquad' "fr"

# QG models
qag_metric 't5-small-squad' 'lmqg/qg_squad' "en"
qag_metric 't5-base-squad' 'lmqg/qg_squad' "en"
qag_metric 't5-large-squad' 'lmqg/qg_squad' "en"
qag_metric 'bart-base-squad' 'lmqg/qg_squad' "en"
qag_metric 'bart-large-squad' 'lmqg/qg_squad' "en"

qag_metric 'mt5-small-dequad' 'lmqg/qg_dequad' "de"
qag_metric 'mt5-small-esquad' 'lmqg/qg_esquad' "es"
qag_metric 'mt5-small-itquad' 'lmqg/qg_itquad' "it"
qag_metric 'mt5-small-jaquad' 'lmqg/qg_jaquad' "ja"
qag_metric 'mt5-small-koquad' 'lmqg/qg_koquad' "ko"
qag_metric 'mt5-small-ruquad' 'lmqg/qg_ruquad' "ru"
qag_metric 'mt5-small-frquad' 'lmqg/qg_frquad' "fr"
