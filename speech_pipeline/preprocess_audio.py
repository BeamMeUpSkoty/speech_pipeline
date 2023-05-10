from scipy.io import wavfile
#import noisereduce as nr
import subprocess


class AudioFile(object):
	"""
	"""
	def __init__(self, PATH, language, num_speakers):
		"""

		Parameters
		-----------
		PATH : string
			path to audio file
		language : string
			two digit code indicating language of audio file
		num_speakers : int
			number of speakers in audio file
		"""
		self.path = PATH
		self.language = language
		self.num_speakers = num_speakers


	def convert_to_wav(self, verbose=False):
		""" uses ffmpeg to convert audio to wav format

		Parameters
		--------
		verbose: boolean, default False
			prints updadtes on converting file 

		Return
		--------
		wav_path: string 
			path to new wav file
		"""
		if verbose:
			print('#### Converting', self.path, 'to WAV file ####')

		wav_path  = self.path[:-4] + '.wav'
		subprocess.call(['ffmpeg', '-i', self.path, wav_path, '-y'])
		self.path = wav_path
		return wav_path

'''
	def reducing_noise(self):
		"""
		"""
		if self.path[-3:] != 'wav':
			self.convert_to_wav()

		rate, data = wavfile.read(self.path)
		# perform noise reduction
		reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=True)
		wavfile.write('preprocessed_' + self.path, rate, reduced_noise)
		return

if __name__ == '__main__':
	PA = PreprocessAudio('adrso002.wav')
	PA.reducing_noise()
'''