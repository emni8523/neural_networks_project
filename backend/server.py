import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
from transformers import AutoTokenizer, pipeline, WhisperFeatureExtractor, WhisperTokenizer, AutoModelForSpeechSeq2Seq
from evaluate import load
from openai import OpenAI
import re
import numpy as np

import tensorflow as tf
from dotenv import load_dotenv


##################################
# import keras
# from keras.layers import Layer
# from keras.saving import register_keras_serializable

# import transformers
from ParagonFull import BertLayer


load_dotenv()
APIKey = os.getenv('APIKEY')
#APIKey = "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"
print(APIKey)

app = Flask(__name__)
CORS(app)



# class putting together Emil's speech to text model and one of the above models, adding in llm calls as well
class Model_init():
    def __init__(self,model,API):
        self.model = model


        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API,
            )

        self.word_error = load("wer")
        self.asr = pipeline(task="automatic-speech-recognition",
            tokenizer=WhisperTokenizer.from_pretrained("json_files/"),
            model=AutoModelForSpeechSeq2Seq.from_pretrained("json_files/"),
            feature_extractor=WhisperFeatureExtractor())
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def predict_sequence(self,raw_text):

        encoding = self.tokenizer.encode_plus(raw_text.split(), 
                                        max_length=40, #est earlier
                                        truncation=True, 
                                        padding="max_length",
                                        is_split_into_words=True,
                                        return_token_type_ids=False, #don't need these
                                        return_tensors='tf' #for tensorflow
                                        )

        print(encoding)
        
        preds = self.model.predict({"input_ids": encoding["input_ids"],
                            "attention_mask": encoding["attention_mask"]})
        

        pred_classes = np.argmax(preds, axis=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        word_ids = encoding.word_ids(batch_index=0)

        pred_words = []
        current_word_tokens = [] 
        current_word_id = None

        for tok, label, wid in zip(tokens,pred_classes,word_ids):
            if wid is None:
                continue

            if wid != current_word_id: # new word
                if current_word_id is not None and first_label == 2:
                    pred_words.append("".join(current_word_tokens))

                current_word_id = wid
                current_word_tokens = [tok.replace("##", "")]
                first_label = label
                
            else: 
                current_word_tokens.append(tok.replace("##",""))
            
        if current_word_id is not None and first_label == 2:
            pred_words.append("".join(current_word_tokens))


        pred_sentence = " ".join(pred_words)
        pred_sentence = re.sub(r' ([,.?!:;])',r'\1',pred_sentence)
        pred_sentence = re.sub(r" ' ", r"'", pred_sentence)
        
        return pred_classes, pred_sentence, encoding

    def transcribe(self, audio):
        transcription = self.asr(audio)["text"].lower()
        return self.pre_process_text(transcription)

    def pre_process_text(self, text):
        text = re.sub(r'([,.?!:;])',r' \1',text)
        return text

    def call_api(self, text):
        content = "Rewrite the following sentence in formal language while preserving its exact meaning and sentiment. Only return the relevant sentence and nothing else: \n" + text
        # API Call
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[{
                    "role": "user",
                    "content": content}
                ]
        )
        
        response = response.choices[0].message.content
        return response

    def run(self, audio):
        transcribed_audio = self.transcribe(audio)
        print(f"Transcribed audio: '{transcribed_audio}'")
        print(type(transcribed_audio))
        
        filtered_text = self.predict_sequence(transcribed_audio)[1]
        print(f"Filtered text: '{filtered_text}'")
        response = self.call_api(filtered_text)

        transcribed_audio = re.sub(r' ([,.?!:;])',r'\1',transcribed_audio)
        transcribed_audio = re.sub(r" ' ", r"'", transcribed_audio)
        
        return transcribed_audio, response



#Loading in the model
print('Loading the transcraption model')

model = tf.keras.models.load_model('model2new.keras', custom_objects={"BertLayer": BertLayer}
                                   )

filter_model = Model_init(model, APIKey)




@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    # Save the uploaded audio
    save_path = 'output/recording.wav'
    os.makedirs('output', exist_ok=True)
    audio_file.save(save_path)
    
    print("Audio received and saved:", save_path)
    
    try:
        print('PASSING ADUIO TO MODEL')
        # Load and preprocess audio
        audio_input, sr = librosa.load(save_path, sr=16000)
        
        # Process with the model
        transcription, final_output = filter_model.run(audio_input)
        print(50 * '-')
        print(transcription)
        print(final_output)
        print(50 * '-')
        
        return jsonify({
            "original_text": transcription,
            "filtered_text": final_output  
        })
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port, host="0.0.0.0")
