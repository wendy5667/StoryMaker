
import os
import six
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import speech_v1p1beta1 as speech
import io
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="vr-project-bf080218f824.json"
def to_wav(speech_file):
    client = speech.SpeechClient()
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.types.RecognitionAudio(content=content)
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code='zh-TW',
        audio_channel_count=2,
        # Enable automatic punctuation
        enable_automatic_punctuation=True)

    response = client.recognize(config, audio)
    result = ""
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        result += str(alternative.transcript)
    return result

def catch_entity(text,entity_list):
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Instantiates a plain text document.
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects entities in the document. You can also analyze HTML with:
    #   document.type == enums.Document.Type.HTML
    entities = client.analyze_entities(document).entities

    for entity in entities:
        entity_type = enums.Entity.Type(entity.type)
        print('=' * 20)
        print(u'{:<16}: {}'.format('name', entity.name))
        print(u'{:<16}: {}'.format('type', entity_type.name))
        print(u'{:<16}: {}'.format('salience', entity.salience))
        print(u'{:<16}: {}'.format('wikipedia_url',
            entity.metadata.get('wikipedia_url', '-')))
        print(u'{:<16}: {}'.format('mid', entity.metadata.get('mid', '-')))

entity_list = []
text = '我喜歡藍色頭髮，不要戴眼鏡，黑人,眼睛'
# text1= "但是我不喜歡紅色的頭髮"
# for t in to_wav('new.wav').split("，"):
#     catch_entity(t,entity_list)
catch_entity(text,entity_list)