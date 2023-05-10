import pyannote.audio
from pyannote.audio import Audio
from pyannote.core import Segment

import contextlib
import wave

from sklearn.cluster import AgglomerativeClustering
import numpy as np


class DiarizeAudio(object):
    """
    """
    def __init__(self, AudioFile, segments, model=None):
        """

        AudioFile : AudioFile Object
          object containing path to audio file, language and number of speakers
        segments : list of dictionaries
          created with whisper model 
        model : load speechbrain 'spkrec-ecapa-voxceleb' model with pyannote
          instantiated in SpeechPipeline Class

        """
        self.path = AudioFile.path
        self.audiofile = AudioFile
        self.segments = segments
        self.embedding_model = model

        #check to see if audio file is in wav format, if not convert to wav format
        if self.audiofile.path[-3:] != 'wav':
            #update self.path to new wav file
            self.path = self.audiofile.convert_to_wav()


    def _segment_embedding(self, segment):
      """ get segment embedding from speechbrain speaker diarization model

      Parameters
      -----------
      segment : dictionary
        segment of audio determined by whisper ASR model.

      Returns
      -----------
      segment_embedding : xxx
        
      """

      #audio processing 
      with contextlib.closing(wave.open(self.path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

      #create pyannote Audio object
      audio = Audio()

      #get start time of segment
      start = segment["start"]
      # Whisper overshoots the end timestamp in the last segment
      end = min(duration, segment["end"])
      #create pyannote segment object
      clip = Segment(start, end)
      #crop audio
      waveform, sample_rate = audio.crop(self.path, clip)
      #get embedding from model
      return self.embedding_model(waveform[None])


    def clustering(self):
      """ gets embedding for each segment. Uses agglomerative clustering to cluster segments 
      based on the number of speakers. Assigns speaker label to each segments (utterance).

      Returns
      -----------
      self.segments : list of dictionaries
        updates self.segments in class with the speaker label from diarization model
      """

      embeddings = np.zeros(shape=(len(self.segments), 192))

      #get segment embedding from diarization model
      for i, segment in enumerate(self.segments):
        embeddings[i] = self._segment_embedding(segment)

      #handle nans
      embeddings = np.nan_to_num(embeddings)

      #using clustering to determine speaker labels
      clustering = AgglomerativeClustering(self.audiofile.num_speakers).fit(embeddings)
      labels = clustering.labels_

      #update self.segments with speaker labels
      for i in range(len(self.segments)):
        self.segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

      return self.segments

