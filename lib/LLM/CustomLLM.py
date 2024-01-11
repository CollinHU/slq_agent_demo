import requests
from http import HTTPStatus

from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class QwenLLM(LLM):
    url: str
    model: str
    token: str
    temperature: float = 0.2
    repetition_penalty: float = 1.1
    topK: int = 50
    topP: float = 0.8
    stop: list = None

    @property
    def _llm_type(self) -> str:
        return "QwenLLMCustom"
    
    def _call(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    )-> str:
        json_request = {
            "url": self.url,
            "headers": {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            "data" :{
                "model": self.model,
                "input":{
                    "prompt": prompt,
                    #"messages": messages
                },
                "parameters":{
                    "stop": self.stop,
                    "temperature" : self.temperature,
                    "repetition_penalty" : self.repetition_penalty,
                    "top_k": self.topK,
                    "top_p": self.topP
                }
            }
        }
        #print(prompt)
        response = requests.post(json_request['url'], headers=json_request['headers'], json=json_request['data'])
        #print(response)
        if response.status_code == HTTPStatus.OK:
            return response.json()['output']['text']
        else:
            print(response.json())
            return response.json()