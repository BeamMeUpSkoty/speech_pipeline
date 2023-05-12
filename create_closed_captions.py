import pandas as pd
from datetime import datetime, timedelta

class CreateClosedCaptions(object):
	"""
	"""
	def __init__(self, filename, segments, outpath):
		""" Transcribe audio with whisper model

		Parameters
		-----------
		filename : string
			name of the file being transcribed
		segments  : list of dictionaries or pandas df
			created in the transcribe_audio_file method of or diarize_audio_file
			method in SpeechPipeline class. Text segments of Audio.
		outpath : string
			path to save vtt file to
		"""

		if isinstance(segments, pd.DataFrame):
			self.segments = segments
		else:
			self.segments = pd.DataFrame(segments)

		self.outpath = outpath
		self.filename = filename

	# define function to convert seconds to desired format
	def convert_seconds(self, seconds):
		"""
		Parameters
		-----------
		seconds : xxx
			xxx

		Returns
		-----------
		formatte_times : xxx
			xxx
		"""

		# create timedelta object with total seconds
		td = timedelta(seconds=seconds)
		# use strftime method to format timedelta object as desired
		return (datetime.min + td).strftime('%H:%M:%S.%f')[:-3]


	def create_vtt_file(self, verbose):
		"""
		Returns
		----------
		vtt_file: file
			saved to location of outpath
		"""

		vtt_path = self.outpath+self.filename
		# Load the CSV file into a pandas dataframe
		
		# Convert the timecode columns to timedelta format
		self.segments['start_vtt'] = self.segments['start'].apply(self.convert_seconds)
		self.segments['end_vtt'] = self.segments['end'].apply(self.convert_seconds)
		
		# Write the WebVTT file
		with open(vtt_path, 'w') as f:
			# Write the WebVTT header
			f.write('WEBVTT\n\n')
			
			# Loop through each row of the dataframe and write the subtitle data
			for index, row in self.segments.iterrows():
				# Write the subtitle index
				f.write(str(index + 1) + '\n')
				
				# Write the subtitle timecode
				f.write(str(row['start_vtt']) + ' --> ' + str(row['end_vtt']) + '\n')
				
				# Write the subtitle text
				f.write(str(row['text']) + '\n\n')

		if verbose:
			print('#### File saved to ', vtt_path, ' ####')
		return