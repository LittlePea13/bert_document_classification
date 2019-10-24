def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
import os
import requests

import urllib.request
from pytorch_transformers.convert_tf_checkpoint_to_pytorch import  convert_tf_checkpoint_to_pytorch
download_file_from_google_drive('1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD', 'biobert_v1.1_pubmed.tar.gz')
os.system('tar -xzf biobert_v1.1_pubmed.tar.gz', 'biobert_v1.1_pubmed/bert_config.json')
convert_tf_checkpoint_to_pytorch('biobert_v1.1_pubmed/model.ckpt-1000000', 'biobert_v1.1_pubmed/pytorch_model.bin' )
os.system('pytorch_transformers bert --tf_checkpoint_path biobert_v1.1_pubmed/model.ckpt-1000000 --bert_config_file biobert_v1.1_pubmed/bert_config.json --pytorch_dump_path biobert_v1.1_pubmed/pytorch_model.bin')
os.system('mv biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/config.json')