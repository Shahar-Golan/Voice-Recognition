============================================================
AUDIO + DIARIZATION + ASR PIPELINE (.prompt)
Using mono file:
C:\Users\golan\VisualStudioProjects\Voice-Recognition\podcast_analysis\data\processed\podcast_16k_mono.wav
============================================================


----------------------------------------------
## 1. Prerequisites (one-time setup)
----------------------------------------------

**🟢 ✅ 1.1. Install additional packages for diarization : COMPLETED**

Tasks:
- In the activated .venv, install:
    pip install pyannote.audio==3.3.1
    pip install torch  (if not already installed)
- Make sure "ffmpeg" is available (already used by whisper, so this should be OK).

Checks:
- In Python REPL:
    >>> import pyannote.audio
    >>> import torch
  No errors.


**🟢 ✅ 1.2. Configure HuggingFace access token (for diarization model) : COMPLETED**

Tasks:
- Set the token as an environment variable (Windows PowerShell example):

    $env:HUGGINGFACE_TOKEN = you can find it in .env

- Optionally, store it permanently in your user env vars through Windows settings.

Checks:
- In Python REPL:
    >>> import os
    >>> os.environ.get("HUGGINGFACE_TOKEN")
  Should print your token string (not None).


----------------------------------------------
## 2. Speaker Diarization on Mono Podcast
----------------------------------------------

Note: At this point, you already have a single mono wav:
- data/processed/podcast_16k_mono.wav (16kHz, ~3 hours)

[✅] 2.1. Implement diarize_podcast in diarization.py

File: podcast_analysis/src/diarization.py

Function: diarize_podcast()

Inputs:
- data/processed/podcast_16k_mono.wav

Outputs:
- outputs/audio_features/diarization_segments.json

Tasks:
- Load the diarization model, e.g.:
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )
- Run the pipeline on the mono audio file:
    diarization = pipeline("data/processed/podcast_16k_mono.wav")
- Convert diarization result into a list of segments with:
    - "speaker": e.g. "S0", "S1", "S2" (use the labels from diarization)
    - "start": float seconds
    - "end": float seconds
- Save to JSON:
    - outputs/audio_features/diarization_segments.json

Example JSON structure:

{
  "audio_file": "data/processed/podcast_16k_mono.wav",
  "segments": [
    {
      "speaker": "S0",
      "start": 1077.16,
      "end": 1078.86
    },
    {
      "speaker": "S1",
      "start": 1078.86,
      "end": 1081.96
    },
    ...
  ]
}

Checks:
- Print total number of diarization segments.
- Print how many unique speakers (e.g. S0, S1).
- Print a few sample segments to verify that speaker labels alternate in a way that makes sense (e.g., S0, S1, S0, S1…).


----------------------------------------------
## 3. ASR on Full Mono Audio (No Fake A/B)
----------------------------------------------

[✅] 3.1. Implement transcribe_podcast in asr_transcript.py

File: podcast_analysis/src/asr_transcript.py

Function: transcribe_podcast()

Inputs:
- data/processed/podcast_16k_mono.wav

Outputs:
- outputs/audio_features/transcript_words.json

Tasks:
- Use openai-whisper or faster-whisper (base or small) to transcribe:
    - Load model once: whisper.load_model("base")  (or "small" for speed/accuracy tradeoff)
- Enable timestamps per segment and per word.
- Run transcription on the mono file.
- Save ASR result as JSON with:
  {
    "audio_file": "data/processed/podcast_16k_mono.wav",
    "sample_rate": 16000,
    "segments": [
      {
        "start": float,
        "end": float,
        "text": "string",
        "words": [
          {"start": float, "end": float, "word": "string"},
          ...
        ]
      },
      ...
    ]
  }

Checks:
- Print total number of ASR segments.
- Print total number of words.
- Print a few sample segments (start/end/text) to verify that timestamps and text look reasonable.
- Confirm language detection is English.


----------------------------------------------
## 4. Merge Diarization + ASR into Speaker-Labeled Transcript
----------------------------------------------

[✅] 4.1. Implement merge_diarization_and_asr in merge_speakers.py

File: podcast_analysis/src/merge_speakers.py

Function: merge_diarization_and_asr()

Inputs:
- outputs/audio_features/diarization_segments.json
- outputs/audio_features/transcript_words.json

Output:
- outputs/audio_features/transcript_with_speakers.json

Goal:
- Combine diarization segments (who spoke when) with ASR word-level timestamps (what was said when) to produce a final transcript with real speaker labels.

Tasks:
- Load diarization_segments.json:
    - list of segments: {"speaker": "S0", "start": ..., "end": ...}
- Load transcript_words.json:
    - list of ASR segments, each with word-level timestamps.
- For each diarization segment:
    - Collect all words whose [start, end] overlap that segment interval.
    - Sort words by start time.
    - Build a text string from those words ("word1 word2 word3 ...").
- For each diarization segment, create a final segment structure:

  {
    "speaker": "S0",
    "start": float,      // diarization start
    "end": float,        // diarization end
    "text": "combined text from ASR words in this interval",
    "words": [
      {"start": float, "end": float, "word": "string"},
      ...
    ]
  }

- Save as:
  outputs/audio_features/transcript_with_speakers.json

Final JSON structure example:

{
  "audio_file": "data/processed/podcast_16k_mono.wav",
  "sample_rate": 16000,
  "speakers": ["S0", "S1"],
  "segments": [
    {
      "speaker": "S0",
      "start": 1077.16,
      "end": 1078.86,
      "text": "And they got knocked out one by one",
      "words": [
        {"start": 1077.16, "end": 1077.46, "word": "And"},
        {"start": 1077.46, "end": 1077.56, "word": "they"},
        ...
      ]
    },
    {
      "speaker": "S1",
      "start": 1078.86,
      "end": 1081.96,
      "text": "But I got to like some of them, some of them I didn't like at all",
      "words": [...]
    },
    ...
  ]
}

Checks:
- Print:
  - number of final segments
  - unique speakers and how many segments each has (e.g., S0: N segments, S1: M segments)
- Manually inspect:
  - A sample of consecutive segments around a known passage (e.g., the “got knocked out one by one” part) and check:
      - The text flows correctly.
      - Speaker labels (S0/S1) switch where you intuitively expect different people to speak.
- Compute approximate speaking time per speaker:
  - For each speaker label, sum (end - start).
  - Print conversational distribution, e.g.:
      - Speaker S0: 52% of time
      - Speaker S1: 48% of time


----------------------------------------------
## 5. Optional: Analysis & Feature Extraction
----------------------------------------------

[ ] 5.1. Use transcript_with_speakers.json for downstream analysis

Tasks:
- Use transcript_with_speakers.json as the single source of truth for:
    - speaking rate (words per minute per speaker)
    - interruptions / overlaps (if diarization supports overlapping segments)
    - topic segmentation (later phase using text embeddings)
- Implement analysis scripts (e.g., in analysis_features.py) that:
    - Load transcript_with_speakers.json
    - Group segments by speaker
    - Compute:
        - total words per speaker
        - words per minute per speaker
        - average segment length
        - etc.

Checks:
- Print a short summary:
    - Speaker S0:
        - total words
        - total speaking time
        - words per minute
    - Speaker S1:
        - same metrics
- Confirm that the metrics align with your intuition based on listening to some parts of the podcast.


============================================================
END OF .PROMPT FILE
============================================================
