import os
import sys
import json

from spacy.cli import download


def download_models():
    languages = [sys.argv[i] for i in range(1,len(sys.argv))]
    if len(languages) == 0:
        languages = os.getenv("languages", "en").split()
        
    print(f"Languages: {languages}")
    for lang in languages:
        print(f"Downloading {lang}")
        download(model=lang, direct=False)

    print("Updating frontend settings...")
    frontend_settings = json.load(open("frontend/_data.json"))

    frontend_settings['index']['languages'] = {
        l: l for l in languages
    }
    frontend_settings['index']['default_language'] = languages[0]

    json.dump(frontend_settings, open("frontend/_data.json", "w"),
              sort_keys=True,
              indent=2)

    print("Done!")
