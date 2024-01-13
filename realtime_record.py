import time
import threading
import sys

import nls
import pyaudio
import json
 
mic = None
stream = None

URL="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
TOKEN="7b234031339b4fbfb77e3b8848d8d94e"  #参考https://help.aliyun.com/document_detail/450255.html获取token
APPKEY="COEXcSikqacV57Rx"    #获取Appkey请前往控制台：https://nls-portal.console.aliyun.com/applist

#messages = []
def print_text(txt):
    print(f'test {txt}')
#以下代码会根据音频文件内容反复进行实时语音识别（文件转写）
class TestSt:
    def __init__(self, tid, print_text, event, messages):
        self.__th = threading.Thread(target=self.__test_run, args=(event,))
        self.__id = tid
        self.__print = print_text
        self.messages = messages
        #self.__stop_event = event
    
    def start(self):
        global mic
        global stream
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)
        self.__th.start()

    def test_on_sentence_begin(self, message, *args):
        print("test_on_sentence_begin:{}".format(json.loads(message)))

    def test_on_sentence_end(self, message, *args):
        sentence = json.loads(message)['payload']['result']
        self.__print(sentence)
        self.messages.append(sentence)

    def test_on_start(self, message, *args):
        print("test_on_start:{}".format(json.loads(message)))

    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(json.loads(message)))

    def test_on_close(self, *args):
        global mic
        global stream
        print('RecognitionCallback close.')
        stream.stop_stream()
        stream.close()
        mic.terminate()
        stream = None
        mic = None
        print("on_close: args=>{}".format(args))

    def test_on_result_chg(self, message, *args):
        self.__print(json.loads(message)['payload']['result'])
        #self.__print(message)

    def test_on_completed(self, message, *args):
        #self.__print(message)
        print("on_completed:args=>{} message=>{}".format(args, message))


    def __test_run(self, event):
        global stream
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
            if stream:
                self.__slices = stream.read(3200, exception_on_overflow = False)
                sr.send_audio(self.__slices)
                time.sleep(0.01)
                sr.ctrl(ex={"test":"tttt"})
            else:
                break
        r = sr.stop()

#def multiruntest(num=500):
#    for i in range(0, num):
#        name = "thread" + str(i)
#        event = threading.Event()
#        t = TestSt(name, print_text, event)
#        t.start()
#        start = time.time()
#        while time.time() - start < 15:
#           continue
#        event.set()
#        time.sleep(3)
#        for msg in messages:
#            print(msg)
#
#nls.enableTrace(False)
#multiruntest(1)