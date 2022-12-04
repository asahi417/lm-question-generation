# Compute QAG score for QG model with gold answer

qag_metric () {
  MODEL=${1}
  DATA=${2}
  LA=${3}
  git clone "https://huggingface.co/lmqg/${MODEL}"
#  lmqg-eval-qag --use-auth-token -m "lmqg/${MODEL}" -e "${MODEL}/eval" -d "${DATA}" --language "${LA}"
  lmqg-push-to-hf -m "${MODEL}" -a "${MODEL}" -o "lmqg"
}

# QG models
qag_metric 't5-small-squad-qg' 'lmqg/qg_squad' "en"
qag_metric 't5-base-squad-qg' 'lmqg/qg_squad' "en"
qag_metric 't5-large-squad-qg' 'lmqg/qg_squad' "en"
qag_metric 'bart-base-squad-qg' 'lmqg/qg_squad' "en"
qag_metric 'bart-large-squad-qg' 'lmqg/qg_squad' "en"

qag_metric 'mbart-large-cc25-jaquad-qg' 'lmqg/qg_jaquad' "ja"
qag_metric 'mbart-large-cc25-esquad-qg' 'lmqg/qg_esquad' "es"
qag_metric 'mbart-large-cc25-dequad-qg' 'lmqg/qg_dequad' "de"
qag_metric 'mbart-large-cc25-itquad-qg' 'lmqg/qg_itquad' "it"
qag_metric 'mbart-large-cc25-koquad-qg' 'lmqg/qg_koquad' "ko"
qag_metric 'mbart-large-cc25-ruquad-qg' 'lmqg/qg_ruquad' "ru"
qag_metric 'mbart-large-cc25-frquad-qg' 'lmqg/qg_frquad' "fr"

qag_metric 'mt5-small-jaquad-qg' 'lmqg/qg_jaquad' "ja"
qag_metric 'mt5-small-esquad-qg' 'lmqg/qg_esquad' "es"
qag_metric 'mt5-small-dequad-qg' 'lmqg/qg_dequad' "de"
qag_metric 'mt5-small-itquad-qg' 'lmqg/qg_itquad' "it"
qag_metric 'mt5-small-koquad-qg' 'lmqg/qg_koquad' "ko"
qag_metric 'mt5-small-ruquad-qg' 'lmqg/qg_ruquad' "ru"
qag_metric 'mt5-small-frquad-qg' 'lmqg/qg_frquad' "fr"

# TODO
qag_metric 'mt5-base-jaquad-qg' 'lmqg/qg_jaquad' "ja"
qag_metric 'mt5-base-esquad-qg' 'lmqg/qg_esquad' "es"
qag_metric 'mt5-base-dequad-qg' 'lmqg/qg_dequad' "de"
qag_metric 'mt5-base-itquad-qg' 'lmqg/qg_itquad' "it"
qag_metric 'mt5-base-koquad-qg' 'lmqg/qg_koquad' "ko"
qag_metric 'mt5-base-ruquad-qg' 'lmqg/qg_ruquad' "ru"
qag_metric 'mt5-base-frquad-qg' 'lmqg/qg_frquad' "fr"
