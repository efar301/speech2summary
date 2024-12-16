from transformers import pipeline
from transcriber import get_transcript
from tqdm import tqdm
import bert_score

def summarize(transcript, length='very_short'):
    
    pipe = pipeline(
    "text2text-generation",
    model="pszemraj/bart-large-summary-map-reduce",
    device_map="auto",
    )
        
    lengths = {
        'very_short': 200,
        'extra_short': 300,
        'short': 400,
        'medium': 500,
        'long': 600,
        'extr_long': 700,
        'very_long': 800
    }
    
    if length not in lengths:
        length = 'medium'
        print(f'Not valid length: defaulting to medium length')
        
    min_length = lengths.get(length)
    
    summary = pipe(
        transcript, 
        max_new_tokens=1024,
        min_length=min_length,
        num_beams=4,
        truncation=True,
        do_sample=True,
        early_stopping=True
    )
    
    return summary[0]['generated_text']

def compound_summarize(t, length='medium', include_metrics=False):
    
    min_chunk_len = 5000
    target_chunk_len = 10000
    overlap = 100
    
    chunk_size = min(target_chunk_len, len(t))
    num_chunks = max(1, (len(t) + chunk_size - overlap - 1) // (chunk_size - overlap))
    chunk_size = max(len(t) // num_chunks, min_chunk_len)
    
    chunks = [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]
    
    for i in range(1, len(chunks)):
        if len(chunks[i]) < min_chunk_len:
            chunks[i - 1] += chunks[i]
            chunks[i] = None
    
    chunks = [chunk for chunk in chunks if chunk is not None]
    
    summaries = [summarize(chunk, length) for chunk in tqdm(chunks)]

    final_summary = ''
    final_summary += final_summary.join(summaries)
    
    if include_metrics:
        summary_len = len(final_summary)
        P, R, F1 = bert_score.score([final_summary], [t], lang='en')
        final_summary += '\n\nMetrics:'
        metrics = f'\n{'precision':<16} = {round(P.item(), 2):<10}{'recall':<16} = {round(R.item(), 2):<10}{'f1':<16} = {round(F1.item(), 2)}'
        final_summary += metrics
        
        stats = f'\n{'text chunk size':<16} = {chunk_size:<10}{'num chunks':<16} = {num_chunks}'
        final_summary += stats
        
        lens = f'\n{'original length':<16} = {len(t):<10}{'summary length':<16} = {summary_len}'
        final_summary += lens
        
        
    return final_summary