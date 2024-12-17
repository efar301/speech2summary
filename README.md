# speech2summary
This project was an endeavor to see how feasible summarizing text from audio is. 
This project went through several iterations and used several models, from Wav2Vec2.0 to WhisperV3 and BART.
Transcription attempts with Wav2Vec2.0 were difficult due to the model's output format being inconsistent with training datasets for summarization models such as BART and Pegasus. 
Eventually, WhisperV3 was chosen to be the transcription model due to its output being similar to the training sets of summarization models. 
In the summarization task, I evaluated several Pegaus and BART models and chose a fine-tuned BART model. 
The project combines WhisperV3 as a transcription model that feeds text into a BART model which summarizes and outputs the text. 
The main drawback of this project is that it relies heavily on the quality of the transcript due to the BART model being highly sensitive to poor input text. 
This project shows the feasibility of using transformer models to transcribe and summarize audio due to the mean BERTScore of 0.85.
