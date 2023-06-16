import subprocess
from pyannote.audio import Audio
from pyannote.core import Segment
import numpy as np 
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import pandas as pd
import warnings
from sklearn.cluster import AgglomerativeClustering
from time import sleep
from sklearn.metrics import calinski_harabasz_score
import math

# TODO: I need to debug the shit out of this .... 


def segment_embedding(start, end, embedding_model, audio_file):
    audio = Audio()
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(audio_file, clip)
    # el embedding_model asigna un vector a cada parte de la transcripcion segun el hablante, sin importar que esta diciendo
    return embedding_model(waveform[None])

def segment_embeddings_divided(segments, segs_per_seg, embeddings, duration, embedding_model, audio_file):
    embeddings = np.zeros(shape=(len(segments)*segs_per_seg,192))
    embedding_index = 0
    for i, segment in enumerate(segments):
        start = segment["start"]
        end = min(duration, segment["end"])
        seg_len = end - start
        for _ in range(segs_per_seg):
            end = start + seg_len/segs_per_seg
            embeddings[embedding_index] = segment_embedding(start, end, embedding_model, audio_file)
            start = end
            embedding_index += 1
    return embeddings

def get_clustering( embeddings,num_speakers=7, num_speakers_auto=False, distance=200):
    if not num_speakers_auto:
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    else:
        clustering = AgglomerativeClustering(n_clusters=None,compute_full_tree=True,distance_threshold=distance).fit(embeddings)
        score = calinski_harabasz_score(embeddings,clustering.labels_)
        # print(score)
    return clustering, score

def get_final_labels(labels_div, segs_per_seg):
    all_speakers = set()
    labels = []
    total_speakers = 0
    for i in range(0,len(labels_div), segs_per_seg):
        print(f"len of labels is {len(labels_div)}")
        label1 = labels_div[i]
        all_equal = True

        for j in range(segs_per_seg):
            if labels_div[i+j] != label1:
                all_equal = False
            if not all_equal:
                labels.append(-10)
            else:
                all_speakers.add(label1)
                labels.append(label1)
                print(f"len(all_speakers) is {len(all_speakers)}")
                # total_speakers += 1
    return labels, len(all_speakers)
    

def get_labels_with_clustering(num_speakers, embeddings, segs_per_seg, num_speakers_auto=False, distance=2000, range_=(3, 4)):

    total_speakers = 0
    
    all_clusterings = []
    best_score = math.inf
    index_best = None
    i = 0
    while total_speakers < range_[0] or total_speakers > range_[1]:
       
        print(f"distance is {distance}")
        print(f"total speakers is {total_speakers}")
        sleep(1)
        clustering, score = get_clustering(embeddings, num_speakers, num_speakers_auto, distance)
        all_clusterings.append({'clustering':clustering,'score':score})
        if score < best_score:
            best_score = score
            index_best = i
        # TODO: FIjarse maniana a partir de esto como elegir el mejor cluster en definitiva. le falta bastante a este codigo. 
                
        labels_div = clustering.labels_
        labels, total_speakers = get_final_labels(labels_div, segs_per_seg)
        if not num_speakers_auto:
            break
        
        if total_speakers < range_[0]: 
            distance /= range_[0] - total_speakers + 1
        elif total_speakers > range_[1]:
            distance *= total_speakers - range_[0] + 1
        i += 1
        
    all_clusterings
           
    labels = np.array(labels)  
    return labels
    
   
def get_pandas(segments, labels):
    # agregamos el speaker segun los labels
    wanted_keys = 'id','start','end','speaker','text' 
    for i,segment in enumerate(segments):
        if labels[i] < 0 :
            segment['speaker'] = "UNKNOWN"
        else:
            segment['speaker'] = f"SPEAKER {labels[i]+1}"
    
   
        # nos deshacemos de los keys que no queremos
        for key in list(segment.keys()):
            if key not in wanted_keys:
                del segment[key]
    
    # organizamos la salida como objeto pandas: DataFrame
    segments_df = pd.DataFrame(segments)
    
    return segments_df

def stereo_to_mono(audio_file):
    sound = AudioSegment.from_file(audio_file)
    if sound.channels > 1:
        sound = sound.set_channels(1)
        sound.export(audio_file, format="wav")

def get_audio_siosi(file, root_dir=None):
    if root_dir is None:
        root_dir = os.getcwd()
        warning_message = "Guardamos los outputs en la carpeta en donde estas localizado"
        warnings.warn(warning_message, Warning)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        
    base_name,ext = os.path.splitext(file)
    if ext == ".wav":
        audio_file = f"{root_dir}/{file}"
    elif ext == ".mp4":
        base_name = file[:-4]
        audio_file = ""
    else:
        audio_file = os.path.join(root_dir, base_name,".wav")
    
    warnings.warn("Espero que me hayas pasado un audio ...", Warning)
    
    # Esta parte es solo para que me tome cualquier formato
    if os.path.exists(audio_file):
        if ext != '.wav':
            wav_file = base_name +'.wav'
            subprocess.call(['ffmpeg', '-i', audio_file, wav_file, '-y'])
            audio_file = wav_file
            
    else:
        
        video = VideoFileClip(os.path.join(root_dir, base_name + ".mp4"))
        audio = video.audio
        audio.write_audiofile(os.path.join(root_dir, base_name,".wav"))
        audio_file = os.path.join(root_dir,base_name,".wav")
        
    # Aqui estoy pasando de estereo a mono, ya que creo que solo funciona con mono
    stereo_to_mono(audio_file) # No necesita retornar nada porque guarda el audio con el mismo nombre, pero en mono ... 
     
    return audio_file # retorna el path completo al audiofile

def format_time(secs):
    # Hours, minutes and seconds
    h, m, s = int(secs // 3600), int(secs % 3600 // 60), int(secs % 60)
    # Milliseconds
    ms = int((secs - int(secs)) * 1000)
    # Format string
    return "{:02}:{:02}:{:02},{:03}".format(h, m, s, ms)

def write_csv(audio_file, segments_df, model_size):
    output_path = audio_file[:-4] + "_" + model_size +  '.csv'
    segments_df.to_csv(output_path)

def create_srt_file(audio_file, model_size, segments_df, duration):
    srt_name = audio_file[:-4] + "_" + model_size + '.srt'

    with open(srt_name, "w") as f_srt:
        for index, segment in segments_df.iterrows():
            start_time = format_time(segment["start"])
            # Assuming the end of the segment is the start of the next one
            end_time = format_time(segment["end"]) if index < len(segments_df) - 1 else format_time(duration)
            
            # Write index
            f_srt.write(str(index+1) + '\n')
            
            # Write timestamps
            f_srt.write(start_time + " --> " + end_time + '\n')

            # Write speaker's name and text
            f_srt.write(segment["speaker"] + ": " + segment["text"][1:] + '\n\n')
            
def convert_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return minutes, round(remaining_seconds)

def print_time(checkpoint, now, last_process):
    seconds = now- checkpoint
    minutes, seconds = convert_seconds(seconds)
    time_message = f"El proceso {last_process} tardo "
    if minutes:
        time_message += f"{minutes} minutos y "
    time_message += f"{seconds} segundos"
    print(time_message)

