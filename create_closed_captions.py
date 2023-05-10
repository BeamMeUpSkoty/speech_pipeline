import pandas as pd
import datetime

class create_closed_captions(object):
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

    def format_timestamps(self);
    	""" format for vtt files should be Hours:Minutes:Seconds:Milliseconds
    	"""
		# Convert the timecode columns to timedelta format
		self.segments['start'] = pd.timestamp(df['start'], unit='s')
		self.segments['end'] = pd.timestamp(df['end'], unit='s')


    	return


	def create_vtt_file(self):
		"""
		Returns
		----------
		vtt_file: file
			saved to location of outpath
		"""

		vtt_path = self.outpath+self.filename
		# Load the CSV file into a pandas dataframe
		
		# Convert the timecode columns to timedelta format
		self.segments['start'] = pd.to_timedelta(df['start'], unit='s')
		self.segments['end'] = pd.to_timedelta(df['end'], unit='s')
		
		# Write the WebVTT file
		with open(vtt_path, 'w') as f:
			# Write the WebVTT header
			f.write('WEBVTT\n\n')
			
			# Loop through each row of the dataframe and write the subtitle data
			for index, row in self.segments.iterrows():
				# Write the subtitle index
				f.write(str(index + 1) + '\n')
				
				# Write the subtitle timecode
				f.write(str(row['start']) + ' --> ' + str(row['end']) + '\n')
				
				# Write the subtitle text
				f.write(str(row['text']) + '\n\n')

		if self.verbose:
			print('#### File saved to ', self.outpath+filename, '_captions.vtt ####')
		return