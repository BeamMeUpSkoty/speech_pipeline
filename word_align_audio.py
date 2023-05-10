import whisperx

class WordAlignAudio(object):
    """
    """
    def __init__(self, audio_file, alignment_model, metadata, device):
        """ Transcribe audio with word-level timestamps. This can only be done
        after running the Whisper model and getting transcribed segments.

        Parameters
        ------------
        audio_file : AudioFile Object
            Audiofile object containing path to audiofile
        alignment_model : whisper model
            whisper model, instantiated in SpeechPipeline Class
        metadata : xxx
            xxx
        device : string
            specifies to use GPU or CPU
        """

        self.audiofile = audio_file
        self.alignment_model = alignment_model
        self.metadata = metadata
        self.device = device

    def align_words(self, segments, verbose=False):
        """ add word-level timestamp alignment to whisper ASR segments

        Parameters
        -----------
        segements : 
        verbose : boolean
            default False, prints statements about alignment progress

        returns
        -----------  
        aligned_segments : 
        """

        if verbose:
            print('### Starting Word-Level Alignment with WhisperX ####')

        return whisperx.align(segments, self.alignment_model, self.metadata, self.audiofile.path, self.device)

