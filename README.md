# Speech Pipeline Documentation

The `speech_pipeline` container handles multilingual ASR, speaker diarization, translation of the ASR to English, word-level alignment for some languages, and can produce a VTT file for use of subtitles in videos.

## Automatic Speech Recognition (ASR) for transcription

To use the `speech_pipeline` container, follow these steps:

### Build Docker

First, build the Docker container using the following command:

```
docker build -t speech_pipeline .
```

### Run Docker interactively

Next, run the Docker container interactively using the following command:

```
docker run -it speech_pipeline
```

### Create Command to Run the speech pipeline

To use the speech pipeline, you can use the command line interface (CLI) built with Fire. The following command can be used to run the pipeline:

```
python3 main.py pipeline
```

#### Required flags

The `pipeline` command requires the following flags:

- `--path`: a string indicating the path to the file that is being transcribed/diarized.
- `--num_speakers`: an integer indicating the number of speakers in the audio file, used for speaker diarization.
- `--language`: a string with a two-character language code indicating the language of the audio being transcribed.

#### Optional flags

The `pipeline` command also supports the following optional flags:

- `--outpath`: a string indicating the path to an output directory. If provided, output files will be saved there.
- `--use_diarize`: a boolean flag indicating whether to run speaker diarization and add speaker labels to the output CSV. Default is `False`.
- `--use_translate`: a boolean flag indicating whether to use the Whisper model to translate text to English. Default is `False`.
- `--use_word_alignment`: a boolean flag indicating whether to produce word alignment rather than segment alignment. Default is `False`.
- `--create_captions`: a boolean flag indicating whether to produce a VTT file with captions for video. Default is `False`.
- `--use_gpu`: a boolean flag indicating whether to check for a GPU and run computations there. Default is `False`.
- `--verbose`: a boolean flag indicating whether to print updates to the console about audio progress. Default is `True`.

## Example Commands

Here are some example commands using the `pipeline` command:

```
python3 main.py pipeline --path 'audio.wav' --num_speakers 2 --language 'en'
```

This command transcribes an audio file located at `'audio.wav'`, assumes there are `2` speakers in the audio file, and the language of the audio is English (`en`).
