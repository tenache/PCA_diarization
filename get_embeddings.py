from time import time
import whisper
import datetime
import subprocess
import  torch
import  pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
import numpy as np 
import os
from moviepy.editor import VideoFileClip
from argparse import ArgumentParser
from pydub import AudioSegment
# from get_foreground import get_foreground



def get_embeddings(num_speakers, language, model_size, segs_per_seg, file):    
    start = time()


    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if language == 'English' and model_size != 'large':
        model_name = model_size + ".en"

    # root_dir = r'D:\PCA_audios'


    # if not file:
    #     base_name = "Transformers"
    #     audio_file = f"{root_dir}/{base_name}.wav"
    # else:
    #     base_name = file[:-4]
    #     if file[-3:] == "wav":
    #         audio_file = os.path.join(root_dir, file)
    #     # elif file[-3:] == "mp4":
    #     #     base_name = file[:-4]
    #     #     audio_file = ""
    #     else:
    #         file_path = os.path.join(root_dir, file)
    #         audio_file = file_path


    # Esta parte es solo para que me tome cualquier formato
    # print(f"audio file is {audio_file}")
    print("\n\n\n")
    file_name, file_ext = os.path.splitext(file)
    if os.path.exists(file):
        if file_ext != 'wav':
            wav_file = file_name + ".wav"
            subprocess.call(['ffmpeg', '-i', file, wav_file, '-y'])
            audio_file = wav_file
           
    else:
        return f"filepath {file} doesn't exist"
        # file_name = os.path.join(root_dir, base_name)
        # video = VideoFileClip(f"{file_name}.mp4")
        # audio = video.audio
        # audio.write_audiofile(f"{root_dir}/{base_name}.wav")
        # audio_file = f"{root_dir}/{base_name}.wav"


    # Aqui estoy pasando de estereo a mono, ya que creo que solo funciona con mono
    def stereo_to_mono(audio_file):
        sound = AudioSegment.from_file(audio_file)
        if sound.channels > 1:
            sound = sound.set_channels(1)
            sound.export(audio_file, format="wav")
        

    stereo_to_mono(audio_file)
    # get_foreground(audio_file)


    # subprocess.call(['spleeter', 'separate', '-p','spleeter:2stems', '-o', root_dir, audio_file])
    # audio_file_fore = f"{root_dir}/{audio_file}/{base_name}/vocals.wav"  
    # Hasta ahora, cargo el modelo solamente
    model = whisper.load_model(model_name)

    print(f"Audio file is {audio_file}")
    time_checkpoint1 = time()
    # Aqui estoy transcribiendo, es decir sacando la parte hablada con whisper
    result = model.transcribe(audio_file)
    print("result is: ")
    print(result)
    print(f"Hasta antes de la transcripcion se hizo en {round(time_checkpoint1 - start)} segundos")
    time_to_transcribe = time() - time_checkpoint1
    print(f"La transcripcion tomo {round(time_to_transcribe)} segundos con el modelo {model_size}")

    # y guardo los segmentos 
    segments = result["segments"]
    print(f"segments is: ")
    print(segments)

    print(f"Len of segments is {len(segments)}")
    # Calcula la duracion del audio
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    print(f"La duracion del video es {duration}")

    audio = Audio()

    def segment_embedding(start, end):
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(audio_file, clip)
        # el embedding_model asigna un vector a cada parte de la transcripcion segun el hablante, sin importar que esta diciendo
        return embedding_model(waveform[None])

    # aqui le estamos asignando un embedding (vector de hablante) a cada segmento que saco whisper
    checkpoint2 = time()


    embeddings = np.zeros(shape=(len(segments*segs_per_seg),192))

    def segment_embeddings_divided(segments, segs_per_seg, embeddings):
        embeddings = np.zeros(shape=(len(segments)*segs_per_seg,192))
        embedding_index = 0
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = min(duration, segment["end"])
            seg_len = end - start
            for _ in range(segs_per_seg):
                end = start + seg_len/segs_per_seg
                embeddings[embedding_index] = segment_embedding(start, end)
                start = end
                embedding_index += 1
        return embeddings

    embeddings = segment_embeddings_divided(segments, segs_per_seg, embeddings)

    embeddings = np.nan_to_num(embeddings)
        
    embeddings_str = []
    for embedding in embeddings:
        embeddings_str.append(str(embedding))
        
    with open("embeddings.txt",'w') as file:
        file.write(str(embeddings_str))
        
    embedding_time = time() - checkpoint2
    print(f"El embedding toma {round(embedding_time)} segundos")

    # supongo que aqui le estamos sacando los nans a los embeddings
    embeddings = np.nan_to_num(embeddings)
    return embeddings, segments, audio_file, duration
    
if __name__ == "__main__":
    # Aca defino el numero de hablantes y el tamano del modelo que quiero utilizar
    parser = ArgumentParser("Argumentos.")
    parser.add_argument("--speakers", default=2, help="Numero de personas que hablan. 2 por default", type=int)
    parser.add_argument("--lang", default="English", help="Idioma en el que esta el video. English por default", type=str)
    parser.add_argument("--size", default="tiny" , help="Tamanio elegido del modelo: tiny, base, small, medium, large", type=str)
    parser.add_argument("--file", default="Life_of_Brian.wav", help="path a archivo de audio o video para ser transformado", type=str)
    parser.add_argument("--division", default=1, help="Un numero alto va a resultar en mas hablantes 'Unknown', pero puede resultar en mejores resultados en gral", type=int)

    args = parser.parse_args()
    print(args)
    
    num_speakers = args.speakers
    language = args.lang
    model_size = args.size
    segs_per_seg = args.division  # numero de segmentos en los cuales dividir cada segmento 
    file = args.file
    get_embeddings(num_speakers, language, model_size, segs_per_seg, file)