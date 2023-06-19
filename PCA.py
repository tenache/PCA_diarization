from time import time
import whisper
import datetime
import subprocess
import torch
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
from sklearn.decomposition import PCA


import pickle as pk
from get_embeddings import get_embeddings
from diarization_limpio import call_diarization

# Lo que queda por hacer es que esto, que esta fiteado con todos los trailers, usarlo para sacar componentes con solamente transform para cada uno de los embeddings que
# queremos utilizar. 

n_components = 96
def get_pca(directory, n_components):
    # print(f"directory is {directory}")
    audio_and_video_f = []
    for dirpath, dirnames, filenames in os.walk(directory):
        # print(f"directory is {directory}")
        for file in filenames:
            print(f"file is {file}")
            file_base, ext = os.path.splitext(file)
            if ext.lower() in [".mp4",".wav",".mp3"]:
                audio_and_video_f.append(os.path.join(dirpath, file))
    all_embeddings = []
    for file_path in audio_and_video_f:
        # print(f"file_path is {file_path}")
        embeddings, segments, audio_file, duration = get_embeddings(8, "English","tiny",1,file_path)
        for embedding in embeddings:
            all_embeddings.append(embedding)
    #OJO: cambiar n_components 
    pca = PCA(n_components=20,copy=True)
    # print(f"all_embeddings is {all_embeddings}")
    # print(f"all_embeddings.shape is {all_embeddings}")
    max_len = 0
    
    #for embedding in all_embeddings:
        # print(f"len of embeddng is {len(embedding)}")
    
    X = np.array(all_embeddings)
    # print(f"X.shape is {X.shape}")

    # print(X.shape)
    #OJO: change components
    pca.fit(X)
    pca_path = os.path.join(directory, "pca.pkl")
    with open(pca_path, "wb") as f:
        pk.dump(pca, f)
    
    temp_path = '/home/thibbard/thibbard/PCA_diarization/pca.pkl'
    with open(temp_path, "wb") as f:
        pk.dump(pca, f)
    # pk.dump(pca,open(pca_path,"wb"))
    # pk.dump(pca,open('/home/thibbard/thibbard/PCA_diarization/pca.pkl',"wb")) # borrar esto despues ... 
        
    """
    # later reload the pickle file
    pca_reload = pk.load(open("pca.pkl",'rb'))
    result_new = pca_reload .transform(X)
    """
if __name__ == "__main__":
    # print('hola')
    # parser = ArgumentParser()
    # parser.add_argument("--dir", default='data', help="directorio donde se encuentran los archivos", type=str)
    # parser.add_argument("--n_components", default=96, help="numero de elementos que tendra el vector de salida por sobre el cual haremos el analisis de aglomeracion", type=str)
    # args = parser.parse_args
    get_pca('audios_2', 96)
    call_diarization('videoplayback.wav', root_dir='/home/thibbard/thibbard/PCA_diarization/audios_2', 
                     model_size='tiny', language="English", num_speakers=7, 
                     segs_per_seg=1, embedding_name="speechbrain/spkrec-ecapa-voxceleb", tell_time=True, num_speakers_auto=True)
    
