import time
import threading
import sys

import nls
import pyaudio
import json
 
from common.utils import Utils
#mic = None
#stream = None

URL="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
TOKEN=Utils.get_alibaba_voice_recognition_token()  #参考https://help.aliyun.com/document_detail/450255.html获取token
APPKEY=Utils.get_alibaba_voice_recognition_appKey()    #获取Appkey请前往控制台：https://nls-portal.console.aliyun.com/applist


#以下代码会根据音频文件内容反复进行实时语音识别（文件转写）
class TestSt:
    def __init__(self, tid, event, messages):
        self.__th = threading.Thread(target=self.__test_run, args=(event,))
        self.__id = tid
        self.__mic = None
        self.__stream = None
        self.messages = messages
        #self.__stop_event = event
    
    def start(self):
        #global mic
        #global stream
        self.__mic = pyaudio.PyAudio()
        self.__stream = self.__mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)
        self.__th.start()

    def test_on_sentence_begin(self, message, *args):
        print("sentence begin:")

    def test_on_sentence_end(self, message, *args):
        sentence = json.loads(message)['payload']['result']
        self.messages.append({'sentence': sentence})
        print("full sentence: ", sentence)

    def test_on_start(self, message, *args):
        print("test_on_start:{}".format(json.loads(message)))

    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(json.loads(message)))

    def test_on_close(self, *args):
        #global mic
        #global stream
        print('RecognitionCallback close.')
        self.__stream.stop_stream()
        self.__stream.close()
        self.__mic.terminate()
        self.__stream = None
        self.__mic = None
        print("on_close: args=>{}".format(args))

    def test_on_result_chg(self, message, *args):
        chunk = json.loads(message)['payload']['result']
        self.messages.append({'chunk': chunk})
        print(chunk)

    def test_on_completed(self, message, *args):
        #self.__print(message)
        print("on_completed:args=>{} message=>{}".format(args, message))
    
    def __test_run(self, event):
        #global stream
        print("thread:{} start..".format(self.__id))
        sr = nls.NlsSpeechTranscriber(
                    url=URL,
                    token=TOKEN,
                    appkey=APPKEY,
                    on_sentence_begin=self.test_on_sentence_begin,
                    on_sentence_end=self.test_on_sentence_end,
                    on_start=self.test_on_start,
                    on_result_changed=self.test_on_result_chg,
                    on_completed=self.test_on_completed,
                    on_error=self.test_on_error,
                    on_close=self.test_on_close,
                    callback_args=[self.__id]
                )
        r = sr.start(aformat="pcm",
                    enable_intermediate_result=True,
                    enable_punctuation_prediction=True,
                    enable_inverse_text_normalization=True)
        while not event.is_set():
            if self.__stream:
                self.__slices = self.__stream.read(3200, exception_on_overflow = False)
                #print('test1')
                sr.send_audio(self.__slices)
                #print('test2')
                time.sleep(0.01)
                sr.ctrl(ex={"test":"tttt"})
            else:
                break
        r = sr.stop()
