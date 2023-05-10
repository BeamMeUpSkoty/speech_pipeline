#import whisperx

class TranscribeAudio(object):
    """
    """
    def __init__(self, audio_file, model):
        """ Transcribe audio with whisper model

        Parameters
        -----------
        Audiofile : AudioFile Object
            Audiofile object containing path to audiofile
        model : whisper model
            whisper model, instantiated in SpeechPipeline Class
        """

        self.audiofile = audio_file
        self.asr_model = model
        #check to see if audio file is in wav format, if not convert to wav format
        if self.audiofile.path[-3:] != 'wav':
            #update self.path to new wav file
            self.audiofile.convert_to_wav()


    def transcribe_audio(self, verbose=False):
        """ transcribe audio

        Parameters
        -----------
        verbose: : boolean, default False
            if true, print statements about progress

        Return
        --------
        segments: list of dictionaries
            translated ASR transcript
        """

        #use whisper model to transcribe
        if verbose:
            print('=== Transcribing',self.audiofile.path,'===')

        result = self.asr_model.transcribe(self.audiofile.path)
        segments = result["segments"]
        return segments
        

    def translate_to_english(self, verbose=False):
        """ translate audio to English

        Return
        --------
        translated ASR transcript
        """
        #use whisper model to transcribe
        if verbose:
            print('=== Translate',self.audiofile.path,'===')

        result = self.asr_model.transcribe(self.audiofile.path, task='translate')
        segments = result["segments"]
        return segments
