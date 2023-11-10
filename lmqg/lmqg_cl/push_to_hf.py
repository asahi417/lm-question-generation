""" Push Models to Modelhub"""
import os
import argparse
import logging
import shutil
from distutils.dir_util import copy_tree
from os.path import join as pj
from glob import glob
from huggingface_hub import Repository

from lmqg.lmqg_cl.readme_template import get_readme
from lmqg import TransformersQG


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Push to Model hub')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    parser.add_argument('-o', '--organization', required=True, type=str)
    parser.add_argument('--use-auth-token', help='Huggingface transformers argument of `use_auth_token`', action='store_true')
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin")), pj(opt.model_checkpoint, "pytorch_model.bin")
    repo_id = f"{opt.organization}/{opt.model_alias}"
    logging.info(f"Upload {opt.model_checkpoint} to {repo_id}")

    model = TransformersQG(opt.model_checkpoint, use_auth_token=opt.use_auth_token)
    model.push_to_hub(repo_id)

    if os.path.exists(opt.model_alias):
        shutil.rmtree(opt.model_alias)
    repo = Repository(opt.model_alias, repo_id)

    # config
    readme = get_readme(model_name=repo_id, model_checkpoint=opt.model_checkpoint)
    with open(pj(opt.model_alias, "README.md"), 'w') as f:
        f.write(readme)

    # upload remaining files
    for i in glob(pj(opt.model_checkpoint, "eval*")):
        copy_tree(i, pj(opt.model_alias, os.path.basename(i)))
    shutil.copyfile(pj(opt.model_checkpoint, "trainer_config.json"), pj(opt.model_alias, "trainer_config.json"))
    repo.push_to_hub()

    # os.system(f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    # shutil.rmtree(opt.model_alias)  # clean up the cloned repo