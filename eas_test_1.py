import os
import json
from typing import Any, Dict, List, Union

from transformers import pipeline

import requests
import gradio as gr
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from fastapi.response import JSONResponse

JSONObject = Dict[str, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

HUGGINGFACE_HUB_CACHE = os.getenv('HUGGINGFACE_HUB_CACHE', '/huggingface')

task = os.getenv('TASK', '')
model_id = os.getenv('MODEL_ID', '')
revision = os.getenv('REVISION', '')
ENABLE_QUEUE = True if os.getenv(
    'ENABLE_QUEUE', 'False').lower() == 'true' else False

MODEL_PATH = os.getenv('MODEL_PATH', '')

if MODEL_PATH != '':
    print("[INFO] using local model")
    model_path = MODEL_PATH
else:
    model_path = HUGGINGFACE_HUB_CACHE + '/models--' + \
        model_id.replace('/', '--') + '/snapshots'
    if not os.path.exists(model_path):
        model_path = model_id
        print("[WARN] local model path not found, downloading from huggingface")
    else:
        snap_shots_list = os.listdir(model_path)
        # sort files by modification time
        snap_shots_list.sort(
            key=lambda x: -os.path.getmtime(model_path + '/' + x))

        find_revision = False
        for snap_shot in snap_shots_list:
            if snap_shot.startswith(revision):
                model_path += '/' + snap_shot
                find_revision = True
                break
        if not find_revision:
            model_path += '/' + snap_shots_list[0]

if task == 'chat':
    import webui_server
    webui_server.main()
else :

    from transformers import AutoTokenizer, AutoModel
    import torch

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Sentences we want sentence embeddings for
    #sentences = ["样例数据-1", "样例数据-2"]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.to(device)
    model.eval()

    # Tokenize sentences
    #encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings

    host = "0.0.0.0"
    port = 8000

    redirect_app = FastAPI()

    @redirect_app.post("/")
    def redirect(request_body: JSONStructure):
        # send request to gradio API
        sentences = request_body['data']
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        try:
            with torch.no_grad():
                model_output = model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            response_data = {'data':sentence_embeddings.tolist()}
        except:
            raise HTTPException(status_code=500, detail='Internal Server Error')
        #print("Sentence embeddings:", sentence_embeddings)
        return JSONResponse(content=response_data)

    app = gr.mount_gradio_app(redirect_app, ss, path="/")
    uvicorn.run(app=app, host=host, port=port)