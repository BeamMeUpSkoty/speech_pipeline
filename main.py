import os

import whisper
#import whisperx

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

import torch
import fire 

import pandas as pd

from transcribe_audio import TranscribeAudio
from diarize_audio import DiarizeAudio
from preprocess_audio import AudioFile
#from word_align_audio import WordAlignAudio
from create_closed_captions import CreateClosedCaptions

class SpeechPipeline(object):

	def __init__(self, path, num_speakers, language, outpath='', use_diarize=False, use_translate=False, use_word_alignment=False, create_captions=False, use_gpu=False, verbose=True):
		"""
		Parameters
		------------
		path : string
			path to directory of audiofiles or single audio file
		num_speakers : int
			number of speakers in the audio file
		language : string
			language of the audio file
		outpath : string
			path to where the transcript and diarization csv will be saved.
		use_diarize : boolean, default False
			if TTrue, will also run speaker diarization and add speaker lable to output csv
		use_translate : boolean, default False
			if True, use the whisper model to translate text to English
		use_word_alignment : boolean, default False
			if True, will produce word aligment rather than segment alignment
		create_captions : boolean, default False
			if True, will produce a vtt file with captions for video.
		use_gpu : boolean, default False
			`if true, then will check if there is a gpu and run computation there
		verbose : boolean, default True
			prints updates to console about audio progress

		Attributes
		---------------
		self.path : string
		self.num_speakers : int
		self.language : string
		self.outpath : string
		self.use_diarize : boolean
		self.use_translate : boolean
		self.use_word_alignment : boolean
		self.create_captions : boolean
		self.verbose : boolean
		self.device : string
		self.whisper_model
		self.alignment_model
		self.metadata
		self.diarization_model
		"""
		self.path = path
		self.num_speakers = num_speakers
		self.language = language
		self.outpath = outpath

		if num_speakers == 1:
			self.use_diarize = False
		else:
			self.use_diarize = use_diarize

		self.use_translate = use_translate
		self.use_word_alignment = use_word_alignment
		self.create_captions = create_captions
		self.verbose = verbose

		if use_gpu:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
			#DECVICE = 'cpu'
		else:
			self.device = 'cpu'

		#load Whisper ASR model
		self.whisper_model = SpeechPipeline.load_whisper_model(use_gpu=use_gpu)

		#load alignment model
		if self.use_word_alignment:
			self.alignment_model, self.metadata = SpeechPipeline.load_alignment_model(language, use_gpu=use_gpu)

		#load speaker_diarization model
		if use_diarize:
			self.diarizaton_model = SpeechPipeline.load_diarization_model(use_gpu=use_gpu)


	@staticmethod
	def load_whisper_model(model_size='base', use_gpu=False):
		""" loads and returns whisper model for multilingual ASR

		Parameters
		------------
		model_size : string
			options from whisper model ['base']
		use_gpu : boolean, default False
			loads model onto gpu if available

		Returns
		----------
		model : whisper model from openAI
		"""

		print('### LOADING WHISPER MODEL ###')

		if use_gpu:
			DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
			#DECVICE = 'cpu'
		else:
			DEVICE = 'cpu'

		try:
			return whisper.load_model('.cache/whisper/'+model_size+'.pt', device=DEVICE)
		except:
			return whisper.load_model(model_size, device=DEVICE)


	@staticmethod
	def load_alignment_model(model_language, use_gpu=False):
		""" loads and returns whisper model for multilingual ASR

		Parameters
		------------
		model_size : string
			options from whisper model ['base']
		use_gpu : boolean, default False
			loads model onto gpu if available

		Returns
		----------
		model : whisper model from openAI
		"""

		print('### LOADING WHISPERX MODEL ###')

		if use_gpu:
			DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
			#DECVICE = 'cpu'
		else:
			DEVICE = 'cpu'

		return whisperx.load_align_model(language_code=model_language, device=DEVICE)


	@staticmethod
	def load_diarization_model(use_gpu=False):
		""" loads and returns speechbrain model for speaker diarization

		Parameters
		------------
		use_gpu : boolean, default False
			loads model onto gpu if available

		Returns
		----------
		model : diarization model from pyannote
		"""

		print('### LOADING DIARIZATION MODEL ###')

		if use_gpu:
			DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			DEVICE = 'cpu'

		return PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=DEVICE)


	def transcribe_audio_file(self, audio_file):
		""" uses the whiper model to transcribe audio. 
		Creates TranscribeAudio object.

		Parameters
		-----------
		audio_file : AudioFile object
			audio file object that needs to be transcribed

		Returns
		-----------
		segments : list of dictionaries
			created with whisper model 

		"""

		if self.verbose:
			print('#### Transcribing Audio at', audio_file.path, '####')

		#create TranscribeAudio class, pass in AudioFile object and ASR model
		TA = TranscribeAudio(audio_file, model=self.whisper_model)

		#call method to transcribe audio
		segments = TA.transcribe_audio(verbose=self.verbose)

		#return transcribed segments
		return segments


	def translate_audio_file(self, audio_file):
		""" uses the whiper model to transcribe audio. 
		Creates TranscribeAudio object.

		Parameters
		-----------
		audio_file : AudioFile object
			audio file object that needs to be transcribed

		Returns
		-----------
		segments : list of dictionaries
			created with whisper model 

		"""

		if self.verbose:
			print('#### Translating and Transcribing Audio at', audio_file.path, '####')

		#create TranscribeAudio class, pass in AudioFile object and ASR model
		TA = TranscribeAudio(audio_file, model=self.whisper_model)

		#call method to transcribe audio
		segments = TA.translate_to_english(verbose=self.verbose)

		#return transcribed segements
		return segments


	def align_to_words_audio_file(self, audio_file, segments):
		"""
		Parameters
		-------------
		audio_file : AudioFile objecty
		segments : list of dictionaries
			created with whisper model 
		vebose : boolean
			default True, prints updates about aligning

		Returns
		-------------
		aligned_segments : list of dictionaries
			with word aligned annotations
		"""

		if self.verbose:
			print('#### Aligning Audio at', audio_file.path, '####')
		
		#create word align audio object
		WAA = WordAlignAudio(audio_file, self.alignment_model, self.metadata, self.device)
		#call method to add word level time stamps
		aligned_segments = WAA.align_words(segments, verbose=self.verbose)
		return aligned_segments

	def diarize_audio_file(self, audio_file, segments):
		"""
		Parameters
		-----------
		audio_file : AudioFile object
			audio file object that needs to be diarized, after transcription
		segments : list of dictionaries
			created in the transcribe_audio_file method of SpeechPipeline class

		Returns
		-----------
		diarized_segments : list of dictionaries
			segments from whisper model ASR with a speaker label
		"""
		if self.verbose:
			print('#### Diarizing Audio at', audio_file.path, '####')

		#create DiarizeAudio class
		DA = DiarizeAudio(audio_file, segments, self.diarizaton_model)
		#call method to assign speaker label to segements
		diarized_segments = DA.clustering()
		#return labeled segments
		return diarized_segments


	def create_transcript(self, filename, segments):
		"""
		Parameters
		-----------
		filename : string
			name of the file being transcribed
		segments : list of dictionaries
			created in the transcribe_audio_file method of or diarize_audio_file
			method in SpeechPipeline class. Text segments of Audio.

		Returns
		-----------
		transcript : csv file 
			csv file of transcript saved at filename_transcript.csv
		"""
		#file_name = os.path.basename(self.path)[:-4]
		#self.file_name = os.path.basename(self.path)[:-4]

		df = pd.DataFrame(segments)
		df.to_csv(self.outpath + filename +'_transcript.csv')

		if self.verbose:
				print('#### File saved to ', self.outpath+filename, '_transcript.csv ####')
		return


	def create_vtt_file(self, filename, segments):
		"""
		Parameters
		-----------
		filename : string
			name of the file being transcribed
		segments  : list of dictionaries
			created in the transcribe_audio_file method of or diarize_audio_file
			method in SpeechPipeline class. Text segments of Audio.

		Returns
		----------
		vtt_file: file
			saved to location of outpath
		"""
		CCC = CreateClosedCaptions(filename, segments, self.outpath)
		CCC.create_vtt_file(self.verbose)
		return

	def pipeline(self):
		""" takes a directory or file. If directory, iterates through all files in the directory. 
		First audio is transcribed with the whisper model. 
		The segments generated by the whisper model are then assigne a speaker label with the speechbrain s
		speaker diarization model. 
		For each audio file, a csv of the transcript is saved to the specificed outpath. 
		"""

		#check to see if path is a directory'
		if os.path.isdir(self.path):
			#get all files in directory
			for file in os.listdir(self.path):
				#files to skip
				if file not in ['.DS_Store']:
					#Make AudioFile object to hold data pertaining to single Audio File
					audio_file = AudioFile(self.path+'/'+file, self.language, self.num_speakers)
					#transcribe audio and return segments for diarization
					segments = self.transcribe_audio_file(audio_file)
					filename = os.path.basename(self.path+'/'+file)[:-4]

					if self.use_word_alignment:
						aligned_segments = self.align_to_words_audio_file(audio_file, segments)
						#outputs csv file with word aligned transcript
						self.create_transcript(filename+'_word_aligned', aligned_segments['word_segments'])

					if self.use_diarize:
						diarized_segments = self.diarize_audio_file(audio_file, segments)
						#outputs transcript of diarized segments
						self.create_transcript(filename+'_diarized', diarized_segments)
					else:
						#outputs transcript of original segments
						self.create_transcript(filename, segments)

					if self.use_translate:
						english_segments = self.translate_audio_file(audio_file)

						if self.use_diarize:
							english_diarized_segments = self.diarize_audio_file(audio_file, english_segments)
							self.create_transcript(filename+'_translated_to_english_diarized', english_diarized_segments)

						else:
							self.create_transcript(filename + '_translated_to_english', english_segments)

					if self.create_captions:
						self.create_vtt_file(filename+'_captions.vtt', segments)



		#if path is a file, this is checked when self.PATH is set in the init for the class
		else:
			audio_file = AudioFile(self.path, self.language, self.num_speakers)
			segments = self.transcribe_audio_file(audio_file)
			filename = os.path.basename(self.path)[:-4]

			if self.use_word_alignment:
				aligned_segments = self.align_to_words_audio_file(audio_file, segments)
				self.create_transcript(filename+'_word_aligned', aligned_segments['word_segments'])

			if self.use_diarize:
				diarized_segments = self.diarize_audio_file(audio_file, segments)
				self.create_transcript(filename+'_diarized', diarized_segments)
			else:
				self.create_transcript(filename, segments)

			if self.use_translate:
				english_segments = self.translate_audio_file(audio_file)

				if self.use_diarize:
					english_diarized_segments = self.diarize_audio_file(audio_file, english_segments)
					self.create_transcript(filename+'_translated_to_english_diarized', english_diarized_segments)

				else:
					self.create_transcript(filename + '_translated_to_english', english_segments)

			if self.create_captions:
				self.create_vtt_file(filename+'_captions.vtt', segments)
		return


if __name__ == '__main__':
	"""
	creates command line interface
	"""
	fire.Fire(SpeechPipeline)
