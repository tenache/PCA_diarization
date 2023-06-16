from time import time
import whisper
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import pickle

import wave
import contextlib
import numpy as np 
from argparse import ArgumentParser

from funciones_auxiliares import segment_embeddings_divided, get_labels_with_clustering, get_pandas
from funciones_auxiliares import write_csv, create_srt_file, get_audio_siosi, print_time

import os
import pandas as pd

def diarization(audio_file, segments, num_speakers=7, segs_per_seg=1, embedding_name="speechbrain/spkrec-ecapa-voxceleb", return_dur=False):
    
    # defino el model de embedding
    embedding_model = PretrainedSpeakerEmbedding(
        embedding_name,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Calcula la duracion del audio porque whisper se pasa en el ultimo segmento y caga todo ... 
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    
    # obtengo los embeddings seguns los segmentos de whisper        
    embeddings = np.zeros(shape=(len(segments*segs_per_seg),192))
    embeddings = segment_embeddings_divided(segments, segs_per_seg, embeddings, duration, embedding_model, audio_file) 
    embeddings = np.nan_to_num(embeddings) # saco los valores no validos
    # lodeamos el modelo pre-entrenado
    
    # OJO: cambiar audios_2
    with open('audios_2/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
     
    # estos embeddings son mas cortos y deberian funcionar mejor 
    embeddings = pca.transform(embeddings)
    print(embeddings.shape)   
    
    # get labels
    # hay que cambiar chooose_num_speakers a false si queremos elegir de antemano
    labels = get_labels_with_clustering(num_speakers, embeddings, segs_per_seg, num_speakers_auto=True)
    segments_df = get_pandas(segments, labels)
    if return_dur:
        return segments_df, duration
    else:
        return segments_df

def call_diarization(audio_file, root_dir=None, 
                     model_size='base', language="English", num_speakers=7, 
                     segs_per_seg=1, embedding_name="speechbrain/spkrec-ecapa-voxceleb", tell_time = False):
    
    start = time()
    if language == 'English' and model_size != 'large':
        model_name = model_size + ".en"
        
    # En esta seccion definimos los modelos, y nos aseguramos de que el audio este en mono y .wav
    
    model = whisper.load_model(model_name)
    audio_file = get_audio_siosi(audio_file, root_dir)
    if tell_time:
        checkpoints = [time()]
        print_time(checkpoints[-1],time(),"cargar modelo y convertir audio")
    
    # obtenemos el resultado de whisper
    segments = model.transcribe(audio_file, verbose=True)["segments"]

    segments_df, duration = diarization(audio_file, segments, num_speakers, segs_per_seg, embedding_name,return_dur=True)

    if tell_time:
        checkpoints.append(time())
        print_time(checkpoints[-1],time(),"diarizacion")
    
    # escribimos un archivo csv
    write_csv(audio_file, segments_df, model_size)
    
    # escribimos un archivo srt
    create_srt_file(audio_file, model_size, segments_df, duration)
    
    if tell_time:
        checkpoints.append(time())
        print_time(checkpoints[-1],time(),"escribir archivos")
        print_time(start, time(), "completo" )
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--audio_file', type=str, required=True, help='Audio file name')
    parser.add_argument('--root_dir', type=str, default=None, help='Root directory')
    parser.add_argument('--model_size', type=str, default='base', help='Model size')
    parser.add_argument('--language', type=str, default='English', help='Language')
    parser.add_argument('--num_speakers', type=int, default=7, help='Number of speakers')
    parser.add_argument('--segs_per_seg', type=int, default=1, help='Segments per segment')
    parser.add_argument('--embedding_name', type=str, default="speechbrain/spkrec-ecapa-voxceleb", help='Embedding name')
    parser.add_argument('--tell_time', type=bool, default=False, help='Tell time it takes to do each thing') 
     
    args = parser.parse_args()

    call_diarization(args.audio_file, args.root_dir, 
                     args.model_size, args.language, args.num_speakers, 
                     args.segs_per_seg, args.embedding_name, args.tell_time)
  
        
        
        
    

    
    

 




    
    
      
    
    
    
    
    
    