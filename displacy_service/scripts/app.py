import torch
import os
import logging

import spacy
from spacy.symbols import ORTH

from wsgiref import simple_server

from displacy_service.server import APP, MODELS, get_model, init_logging

logging.getLogger().setLevel(logging.INFO)

def run():
    if not spacy.prefer_gpu():
        logging.getLogger().warn("Unable to get GPU")

    for model in MODELS:
        logging.info(f"Load model {model}")
        loaded_model = get_model(model)
        special_cases_str = os.getenv(f"{model}_special_cases", "")
        if special_cases_str:
            for special_case in special_cases_str.split(','):
                loaded_model.tokenizer.add_special_case(
                    special_case,
                    [{ORTH: special_case}]
                )

    logging.info("Loaded all models. Starting HTTP server.")
    init_logging()
    httpd = simple_server.make_server('0.0.0.0', 8000, APP)
    httpd.serve_forever()
