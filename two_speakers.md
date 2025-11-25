----------------------------------------------
## 2. Preprocess audio (speaker-separated WAVs)
----------------------------------------------

[ ] 2.1. Implement split_stereo_to_speakers in audio_preprocess.py

Tasks:
- Load data/raw/podcast.mp3 using pydub.AudioSegment.from_file(...)
- Confirm the audio is stereo (channels == 2). If not, raise a warning.
- Split stereo into two mono tracks using audio.split_to_mono():
    - left_channel  -> Speaker A
    - right_channel -> Speaker B
- For each channel:
    - Convert to mono (1 channel)
    - Resample to SAMPLE_RATE (e.g., 16kHz) using .set_frame_rate(SAMPLE_RATE)
- Export as:
    - data/processed/podcast_speaker_A_16k_mono.wav
    - data/processed/podcast_speaker_B_16k_mono.wav

Checks:
- Print final sample rate and channels for each speaker file
- Print duration (seconds/minutes) for each speaker
- Clearly document mapping: "Speaker A = left channel", "Speaker B = right channel"


----------------------------------------------
## 3. Generate transcript with timestamps (ASR)
----------------------------------------------

Note: These transcripts are needed for speaking-rate calculations AND to keep speakers separated.

[ ] 3.1. Implement ASR in asr_transcript.py (per speaker)

Tasks:
- Use openai-whisper or faster-whisper to transcribe:
    - data/processed/podcast_speaker_A_16k_mono.wav
    - data/processed/podcast_speaker_B_16k_mono.wav
- Enable timestamps per segment and (if possible) per word
- For each segment, include:
    {
      "speaker": "A" or "B",
      "start": float,
      "end": float,
      "text": string,
      "words": [
        {"start": float, "end": float, "word": string},
        ...
      ]
    }

Outputs:
- outputs/audio_features/transcript_segments_for_audio.json
  Structure:
  {
    "sample_rate": 16000,
    "speakers": ["A", "B"],
    "segments": [
      {...}, {...}, ...
    ]
  }

Checks:
- Print number of segments per speaker (A and B)
- Print total number of segments
- Inspect a few entries to confirm speaker labeling is correct
