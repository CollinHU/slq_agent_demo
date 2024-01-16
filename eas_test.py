import os
import json
from typing import Any, Dict, List, Union

from transformers import pipeline

import requests
import gradio as gr
from fastapi import FastAPI, HTTPException, Request
import uvicorn

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

    inf_pipeline = pipeline(task, model_path)

    ss = gr.Interface.from_pipeline(inf_pipeline)
    ss.title = 'PAI-EAS: ' + ss.title

    host = "0.0.0.0"
    port = 8000

    redirect_app = FastAPI()

    @redirect_app.post("/")
    def redirect(request_body: JSONStructure):
        # send request to gradio API
        response = requests.post(
            url="http://{0}:{1}/run/predict".format(host, port),
            json=request_body
        )
        if not response.ok:
            raise HTTPException(
                status_code=response.status_code,
                detail=json.loads(response.content))
        response_body = json.loads(response.content)
        return response_body

    app = gr.mount_gradio_app(redirect_app, ss, path="/")
    uvicorn.run(app=app, host=host, port=port)