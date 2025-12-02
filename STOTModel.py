from ParagonFull import*

# class putting together Emil's speech to text model and one of the above models, adding in llm calls as well
class STOT():
    def __init__(self,model,API,audio_data,audio_predictions=[]):
        self.model = model

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API,
            )

        self.word_error = load("wer")
        self.asr = pipeline(task="automatic-speech-recognition",
            tokenizer=WhisperTokenizer.from_pretrained("Model"),
            model=AutoModelForSpeechSeq2Seq.from_pretrained("Model"),
            feature_extractor=WhisperFeatureExtractor())

        self.lower = self.model.lower
        self.segmented = self.model.segmented
        self.audio_predictions=audio_predictions

        self.audio_data = audio_data
        

    # takes in audio data and returns the transcribed text
    def transcribe(self, audio):
        transcription = self.asr(audio)["text"].lower() if self.lower else self.asr(audio)["text"]
        return self.model.pre_process_text(transcription) if self.model.segmented else transcription


    # get more academic sentence
    def call_api(self, text):
        content = "Give me a more academic way of saying the following sentence. Only return the relevant sentence and nothing else: \n" + text
        # API Call
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[{
                    "role": "user",
                    "content": content}
                ]
        )
        
        response = response.choices[0].message.content
        #print(f"New Sentence: {response}")
        return response


    # eval a single audio input
    def eval_single_input(self, num):
        audio = self.audio_data[num]
        unfiltered_transcript = self.model.X_test[num]
        filtered_transcript = self.model.y_test[num]

        transcribed_audio = self.transcribe(audio)
        print(f"Audio Transcription: {transcribed_audio}")
        print(f"True Transcription: {unfiltered_transcript}")
        print(f"Word Error Rate: {self.word_error.compute(predictions=transcribed_audio, references=unfiltered_transcript)}")
        print("-------------------")

        classes_from_model, transcript_from_model, mod_enc = self.model.predict_sequence(transcribed_audio)
        classes_from_data, transcript_from_data, data_enc = self.model.predict_sequence(unfiltered_transcript)

        print(f"Transcription After Filter: {transcript_from_model}")
        print(f"Filtered Gold Text: {transcript_from_data}")
        print(f"True Filter: {filtered_transcript}")

        # get true labels
        if self.model.model_type == "transformer":
            true_enc_mod = self.model.encode_example(transcribed_audio, filtered_transcript)
            true_labels_mod = true_enc_mod["labels"]
            true_enc_data = self.model.encode_example(unfiltered_transcript, filtered_transcript)
            true_labels_data = true_enc_data["labels"]
        else:
            _, true_labels_mod = self.model.encode_example(transcribed_audio, filtered_transcript)
            _, true_labels_data = self.model.encode_example(unfiltered_transcript, filtered_transcript)

        # mask 0s
        mask_model = (true_labels_mod != 0)
        classes_from_model = classes_from_model[mask_model]
        true_labels_mod = true_labels_mod[mask_model]

        mask_data = (true_labels_data != 0)
        classes_from_data = classes_from_data[mask_data]
        true_labels_data = true_labels_data[mask_data]

        # print accuracies
        accuracy_mod = np.mean(classes_from_model == true_labels_mod)
        print(f"Total Model Accuracy: {accuracy_mod}, F1: {f1_score(classes_from_model,true_labels_mod)}")

        accuracy_dat = np.mean(classes_from_data == true_labels_data)
        print(f"Filter Model Accuracy: {accuracy_dat}, F1: {f1_score(classes_from_data,true_labels_data)}")

        # prompt the LLM to get a more academic response
        #self.call_api(transcript_from_model)
        

    # eval entire dataset
    def eval(self):
        # get data
        audio_data = self.audio_data
        intermediary_data = self.model.X_test
        final_data = self.model.y_test

        # transcribe audio if not already transcribed
        if len(self.audio_predictions) == 0:
            for i in range(len(audio_data)):
                audio_pred = self.transcribe(audio_data[i])
                self.audio_predictions.append(audio_pred)

        # get word error rate
        average_wer = self.word_error.compute(predictions=self.audio_predictions, references=intermediary_data)
        print(f"Average Word Error Rate of Stot Model: {average_wer}")
        
        # measure accuracy of filter
        if self.model.model_type == "transformer":
            encoding = [self.model.encode_example(raw, clean) for raw, clean in zip(self.audio_predictions, final_data)]
            enc_ds = Dataset.from_list(encoding)

            enc_tf = tf.data.Dataset.from_tensor_slices((
                {
                    "input_ids": np.array(enc_ds["input_ids"], dtype=np.int32),
                    "attention_mask": np.array(enc_ds["attention_mask"], dtype=np.int32)
                },
                np.array(enc_ds["labels"]),
            ))

            loss, accuracy = self.model.model.evaluate(enc_tf.batch(self.model.batch))

            self.model.print_cm(enc_tf)

        else:
            filter_input, filter_output = self.model.create_binary_dataset(self.audio_predictions, final_data)
            loss, accuracy = self.model.model.evaluate(filter_input, filter_output)

            self.model.print_cm(filter_input, filter_output)

        print(f"Final Accuracy: {accuracy}, Final Loss: {loss}")
    
    def run(self, audio):
        transcribed_audio = self.transcribe(audio)
        filtered_text = self.model.predict_sequence(transcribed_audio)[1]
        response = self.call_api(filtered_text)

        transcribed_audio = re.sub(r' ([,.?!:;])',r'\1',transcribed_audio)
        transcribed_audio = re.sub(r" ' ", r"'", transcribed_audio)
        
        return transcribed_audio, response