# PARAGON
Paragon is a speech to text application which allows users to generate academic text from raw audio. Through a fine-tuned Whisper speech-to-text model, the speech is processed and transcribed into text. Then, through a filtration transformer model pre-trained on BERT, the transcribed text is updated to remove repetition and filler words. We utilize OpenAI’s gpt-oss-20b as an LLM partner to recommend more academic ways of saying the sentence before printing the final result to a final frontend. We report a word error rate for our speech to text model of 9.63% (13.3  points lower than baseline) and an accuracy of 96.6% and an F1 score of 85.2% (comparable to baseline). 

Project description: https://www.youtube.com/watch?v=oahkBf5Fa2I
