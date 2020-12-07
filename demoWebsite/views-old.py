from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.contrib import auth
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
import wave
import cgi
import base64
import io
import os
import wave
import sys
import json
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
from google.cloud import speech_v1p1beta1 as speech
sys.path.append('/home/b05505052/projects/vr/vr-project/demoWebsite/ffmpeg-4.2.2/')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="vr-project-bf080218f824.json"
import pydub
from pydub import AudioSegment
import soundfile
import requests
from demoWebsite.wordbase import face_color, eye_color, hair_color, face_type, nega
import random
from demoWebsite.test import create_img
@csrf_exempt
def main(request):
    data = {"img_url":None}
    if request.POST:
        print("hihihihihi")
        with open('qwer.wav', 'wb') as f_vid:
            f_vid.write(request.body)
        wav_audio = AudioSegment.from_file("qwer.wav")
        wav_audio.export("audio1.wav", format="wav")
        data, samplerate = soundfile.read('audio1.wav')
        soundfile.write('audio2.wav', data, samplerate, subtype='PCM_16')
        content = to_wav("audio2.wav")
        contents = analyze(content)
        print(contents)
        tags = encodeJSON(contents)
        img_path = create_img(tags)
        data = {"img_url": img_path[0]}
        return render(request, 'result.html', data)
        print(data)
    return render(request, 'main.html', data)

def to_wav(speech_file):
    client = speech.SpeechClient()
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.types.RecognitionAudio(content=content)
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code='zh-TW',
        audio_channel_count=1,
        # Enable automatic punctuation
        enable_automatic_punctuation=True)

    response = client.recognize(config, audio)
    output = ""
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        output += str(alternative.transcript)
    return output
def analyze(content):
    result = []
    contents = content.split("，")
    for i in contents:
        request = requests.get('https://westus.api.cognitive.microsoft.com/luis/v2.0/apps/ce1e015f-86e4-44fa-ae0a-404207bda4d6?verbose=true&timezoneOffset=0&subscription-key=681fdaec0ebe49758322f6b66c9180ce&q='+ i)
        result.append(json.loads(request.text))
    return(result)
def encodeJSON(infos):
    hair = [0, 0, 0, 0, 0, 0]
    eye = [0, 0, 0, 0]
    face = [0, 0, 0]
    glasses = [0, 0]
    
    for info in infos:
        if info["topScoringIntent"]["intent"] == "None" or info["topScoringIntent"]["score"] < 0.3:
            continue
        if info["topScoringIntent"]["intent"] == "VR.FaceColor":
            neg_flag = False
            set_color = -1
            for ent in info["entities"]:
                if ent["entity"] in face_color:
                    set_color = face_color[ent["entity"]]
                elif ent["entity"] in face_type:
                    set_color = face_type[ent["entity"]]
                elif ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            if set_color == -1:
                continue
            else: 
                if neg_flag:
                    face[set_color] = -1
                else: 
                    face[set_color] = 1

        elif info["topScoringIntent"]["intent"] == "VR.EyeColor":
            neg_flag = False
            set_color = -1
            for ent in info["entities"]:
                if ent["entity"] in eye_color:
                    set_color = eye_color[ent["entity"]]
                elif ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            if set_color == -1:
                continue
            else: 
                if neg_flag:
                    eye[set_color] = -1
                else: 
                    eye[set_color] = 1
        elif info["topScoringIntent"]["intent"] == "VR.HairColor":
            neg_flag = False
            set_color = -1
            for ent in info["entities"]:
                if ent["entity"] in hair_color:
                    set_color = hair_color[ent["entity"]]
                elif ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            if set_color == -1:
                continue
            else: 
                if neg_flag:
                    hair[set_color] = -1
                else: 
                    hair[set_color] = 1
        elif info["topScoringIntent"]["intent"] == "VR.Glasses":
            neg_flag = False
            for ent in info["entities"]:
                if ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            
            if neg_flag:
                glasses[1] = 1
            else: 
                glasses[0] = 1
        else:
            sys.exit("Function not found")

    features = []
    glasses_ = glasses
    for n in range(6):
        if 1 in face:
            idx = face.index(1)
        else: 
            idx = random.choice([i for i in range(len(face)) if face[i]!=-1])

        face_ = [1 if i == idx else 0 for i in range(len(face))]
        if 1 in eye:
            idx = eye.index(1)
        else: 
            idx = random.choice([i for i in range(len(eye)) if eye[i]!=-1 ])
        eye_ = [1 if i == idx else 0 for i in range(len(eye))]
        if 1 in hair:
            idx = hair.index(1)
        else: 
            idx = random.choice([i for i in range(len(hair)) if hair[i]!=-1 ])
        hair_ = [1 if i == idx else 0 for i in range(len(hair))]
        
        if 1 not in glasses_:
            idx = random.choice([i for i in range(len(glasses))])
            glasses_ = [1 if i == idx else 0 for i in range(len(glasses))]
        else:
            glasses_ = glasses
            
        feature = hair_+eye_+face_+glasses_
        print(feature)
        features.append(feature)
    return features


# if __name__ == '__main__':
# 	import json
# 	input_file = open ('myJSONArray.json',encoding="utf-8")
# 	json_array = json.load(input_file)
# 	code = encodeJSON(json_array)
# 	print(code)