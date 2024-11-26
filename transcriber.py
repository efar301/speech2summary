from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import yt_dlp
import textwrap

def get_transcript(url):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Wav2Vec2ForCTC.from_pretrained(
        'facebook/wav2vec2-large-robust-ft-libri-960h', 
        torch_dtype=torch.float16
    ).to(device)

    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-robust-ft-libri-960h')  
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    cc = ''

    input_audio, _ = librosa.load('audio.mp3', sr=16000)
    input_values = processor(input_audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(torch.float16).to(device)

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
    transcription = processor.batch_decode(predicted_ids)[0]
    cc += transcription 
    cc += ' '
    
    return cc