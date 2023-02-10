""" Push Models to Modelhub"""
import os
import argparse
import logging
import shutil
from distutils.dir_util import copy_tree
from os.path import join as pj
from huggingface_hub import create_repo

import transformers
from lmqg.lmqg_cl.readme_template import get_readme

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
gitattribute = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""


def main():
    parser = argparse.ArgumentParser(description='Push to Model hub')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    parser.add_argument('-o', '--organization', required=True, type=str)
    parser.add_argument('--use-auth-token', help='Huggingface transformers argument of `use_auth_token`',
                        action='store_true')
    parser.add_argument('--skip-model-upload', help='', action='store_true')
    parser.add_argument('--access-token', default=None, type=str)
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin")), pj(opt.model_checkpoint, "pytorch_model.bin")
    logging.info(f"Upload {opt.model_checkpoint} to {opt.organization}/{opt.model_alias}")

    # url = create_repo(opt.model_alias, organization=opt.organization, exist_ok=True)
    create_repo(repo_id=f"{opt.organization}/{opt.model_alias}", exist_ok=True, repo_type="model")

    if not opt.skip_model_upload:
        tokenizer = transformers.AutoTokenizer.from_pretrained(opt.model_checkpoint, local_files_only=True)
        config = transformers.AutoConfig.from_pretrained(opt.model_checkpoint, local_files_only=True)
        if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
            model_class = transformers.T5ForConditionalGeneration.from_pretrained
        elif config.model_type == 'mt5':
            model_class = transformers.MT5ForConditionalGeneration.from_pretrained
        elif config.model_type == 'bart':
            model_class = transformers.BartForConditionalGeneration.from_pretrained
        elif config.model_type == 'mbart':
            model_class = transformers.MBartForConditionalGeneration.from_pretrained
        else:
            raise ValueError(f'unsupported model type: {config.model_type}')

        if opt.access_token is not None:
            model = model_class(opt.model_checkpoint, config=config, local_files_only=True)
        else:
            model = model_class(opt.model_checkpoint, config=config, local_files_only=True)
        args = {"repo_id": f"{opt.organization}/{opt.model_alias}", "use_auth_token": opt.use_auth_token}
        model.push_to_hub(**args, hub_token=opt.access_token)
        tokenizer.push_to_hub(**args)
        config.push_to_hub(**args)

    # upload remaining files
    copy_tree(f"{opt.model_checkpoint}", f"{opt.model_alias}")

    # config
    readme = get_readme(
        model_name=f"{opt.organization}/{opt.model_alias}",
        model_checkpoint=opt.model_checkpoint
    )
    with open(pj(opt.model_checkpoint, "README.md"), 'w') as f:
        f.write(readme)

    # upload remaining files
    copy_tree(opt.model_checkpoint, opt.model_alias)
    with open(f"{opt.model_alias}/.gitattributes", 'w') as f:
        f.write(gitattribute)
    os.system(
        f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(opt.model_alias)  # clean up the cloned repo
