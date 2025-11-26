============================================================
CONVERSATION DYNAMICS ANALYSIS (.prompt)
Input file: outputs/audio_features/transcript_with_speakers.json
============================================================

Assumed JSON structure (per segment):
{
  "audio_file": "data/processed/podcast_16k_mono.wav",
  "sample_rate": 16000,
  "segments": [
    {
      "speaker": "Joe Rogan" | "Donald Trump" | ...,
      "start": float,        // seconds
      "end": float,          // seconds
      "text": string,
      "words": [
        {"start": float, "end": float, "word": string},
        ...
      ]
    },
    ...
  ]
}


----------------------------------------------
## 4. Conversational Stats per Speaker
----------------------------------------------

Goal:
Implement basic, global stats per speaker:
- total speaking time (1.1)
- average words per minute (2.2)

[x] 4.1. Implement basic_speaker_stats in analysis_speaking_features.py

File:
- podcast_analysis/src/analysis_speaking_features.py

Function:
- basic_speaker_stats(
      transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
      output_path: str = "outputs/audio_features/basic_speaker_stats.json"
  ) -> dict

Tasks:
- Load transcript_with_speakers.json.
- For each segment:
  - duration = end - start
  - speaker = segment["speaker"]
  - num_words = len(segment["words"])
- Aggregate per speaker:
  - total_speaking_time_sec = sum(duration)
  - total_words = sum(num_words)
  - total_speaking_time_min = total_speaking_time_sec / 60.0
  - words_per_minute = total_words / total_speaking_time_min  (if time > 0)
  - num_segments = count of segments

Output JSON structure:
{
  "audio_file": "...",
  "speakers": {
    "Joe Rogan": {
      "total_speaking_time_sec": float,
      "total_speaking_time_min": float,
      "total_words": int,
      "words_per_minute": float,
      "num_segments": int
    },
    "Donald Trump": {
      ...
    }
  }
}

Checks:
- Print a summary table to console, e.g.:
    Speaker         Time(min)   Words   WPM   Segments
    Joe Rogan       85.3        24000   281   2100
    Donald Trump    89.2        30000   337   2600
- Verify total time across speakers is close to total conversation duration.
- Save basic_speaker_stats.json to the specified output path.


----------------------------------------------
## 5. Time-Series Speaking Rate per Speaker (2.6)
----------------------------------------------

Goal:
Compute a time series of speaking rate (word count over time), per speaker,
and save both:
- machine-readable data
- a simple time series plot (words vs time) to visualize “energy” over time.

Definitions:
- Use fixed-size time windows over the full conversation.
- Example: window_size_sec = 30.0 (configurable).
- For each window and speaker, count how many words fall inside.

[x] 5.1. Implement speaking_rate_timeseries in analysis_speaking_features.py

Function:
- speaking_rate_timeseries(
      transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
      output_data_path: str = "outputs/audio_features/speaking_rate_timeseries.json",
      output_plot_path: str = "outputs/audio_features/speaking_rate_timeseries.png",
      window_size_sec: float = 30.0,
      step_size_sec: float | None = None
  ) -> dict

Tasks:
- Load transcript_with_speakers.json and determine:
  - conversation_start = min(segment.start)
  - conversation_end   = max(segment.end)
- Define time windows:
  - If step_size_sec is None: use non-overlapping windows (step_size_sec = window_size_sec).
  - Else: use sliding windows [t, t + window_size_sec) with step_size_sec shift.
- For each window and each speaker:
  - Count words whose word["start"] is within [window_start, window_end).
  - Optionally compute words_per_minute = (word_count / window_size_sec) * 60.
- Produce a list of entries:
  - For non-overlapping windows:
    [
      {
        "window_index": int,
        "window_start": float,
        "window_end": float,
        "speaker": string,
        "word_count": int,
        "words_per_minute": float
      },
      ...
    ]

Output JSON structure:
{
  "audio_file": "...",
  "window_size_sec": float,
  "step_size_sec": float,
  "timeseries": [
    {
      "window_index": 0,
      "window_start": float,
      "window_end": float,
      "speaker": "Joe Rogan",
      "word_count": int,
      "words_per_minute": float
    },
    {
      "window_index": 0,
      "window_start": float,
      "window_end": float,
      "speaker": "Donald Trump",
      "word_count": int,
      "words_per_minute": float
    },
    ...
  ]
}

Plot:
- Using matplotlib:
  - X-axis: window center time (e.g., (window_start + window_end)/2).
  - Y-axis: words_per_minute.
  - One line per speaker (e.g., different colors).
  - Save plot to output_plot_path (PNG).

Checks:
- Print basic info:
    - number of windows
    - list of speakers
    - a few sample entries from timeseries.
- Verify that words summed across windows are close to total_words from basic_speaker_stats (allowing small differences at edges).
- Open the PNG and visually confirm that the time series makes sense (peaks in busy talking sections).


----------------------------------------------
## 6. Interruptions & Overlap (merged idea from section 3)
----------------------------------------------

Goal:
Detect meaningful interruptions between speakers.
An interruption should:
- involve overlap between two different speakers
- be long enough (not just short backchannel like “yeah”, “uh-huh”)
- capture who interrupted whom.

Definitions:
- Segment A: [start_A, end_A, speaker_A]
- Segment B: [start_B, end_B, speaker_B]
- If speaker_B != speaker_A and start_B < end_A:
    - overlap_duration = min(end_A, end_B) - start_B
    - If overlap_duration >= MIN_OVERLAP_SEC: candidate interruption.
- Also treat “instant takeover” with very small gap as interruption:
    - If speaker_B != speaker_A and 0 <= (start_B - end_A) <= MAX_GAP_SEC: candidate interruption.
- To filter out trivial one-word overlaps:
    - For the “interrupter” (speaker_B in overlap case):
      - count words in the overlapping / immediate segment.
      - Require at least MIN_WORDS_INTERRUPTER (e.g., 3) to consider it a real interruption.

Constants (tunable, define at top of file):
- MIN_OVERLAP_SEC = 0.2
- MAX_GAP_SEC = 0.15
- MIN_WORDS_INTERRUPTER = 3
- MAX_BACKCHANNEL_DURATION_SEC = 0.6  (short overlaps under this AND short text may be ignored or marked as backchannel).

[x] 6.1. Implement detect_interruptions in analysis_speaking_features.py

Function:
- detect_interruptions(
      transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
      output_path: str = "outputs/audio_features/interruptions.json",
      min_overlap_sec: float = 0.2,
      max_gap_sec: float = 0.15,
      min_words_interrupter: int = 3,
      max_backchannel_duration_sec: float = 0.6
  ) -> dict

Tasks:
- Load transcript_with_speakers.json.
- Sort all segments by start time.
- Iterate over pairs of consecutive segments (i, i+1) and possibly look ahead one segment if needed.
- For each boundary where speaker changes:
  - Compute:
    - overlap_duration = max(0.0, min(end_prev, end_next) - max(start_prev, start_next))
    - gap = start_next - end_prev
  - Case 1: overlap (candidate hard interruption):
    - if overlap_duration >= min_overlap_sec:
      - interrupter = segment with later start (the one that came in while the other speaking).
  - Case 2: near-zero gap (candidate soft interruption):
    - if 0 <= gap <= max_gap_sec:
      - interrupter = next speaker (segment i+1).
  - For the interrupter:
    - count words in that segment.
    - If num_words < min_words_interrupter AND segment_duration < max_backchannel_duration_sec:
      - mark as “backchannel” (ignore or track separately).
    - Else:
      - record as real interruption.

Output JSON structure:
{
  "audio_file": "...",
  "parameters": {
    "min_overlap_sec": ...,
    "max_gap_sec": ...,
    "min_words_interrupter": ...,
    "max_backchannel_duration_sec": ...
  },
  "interruptions": [
    {
      "time": float,                    // time of interruption (e.g., start of interrupter segment)
      "overlap_duration": float,        // may be 0 for soft interruption
      "gap": float,                     // may be negative for overlap
      "interrupter": string,            // speaker who interrupted
      "interrupted": string,            // speaker who was speaking before
      "interrupter_segment_index": int, // index in sorted segments
      "interrupted_segment_index": int, // index in sorted segments
      "interrupter_segment_text": string,
      "interrupted_segment_text": string
    },
    ...
  ],
  "backchannels": [
    {
      "time": float,
      "speaker": string,
      "segment_index": int,
      "text": string
    },
    ...
  ],
  "stats": {
    "total_interruptions": int,
    "total_backchannels": int,
    "per_speaker": {
      "Joe Rogan": {
        "interruptions_made": int,
        "interruptions_received": int,
        "backchannels_made": int
      },
      "Donald Trump": {
        ...
      }
    }
  }
}

Checks:
- Print a short summary:
    - total_interruptions
    - total_backchannels
    - interruptions per speaker (made/received)
- Print a few sample interruptions with:
    - who interrupted whom
    - time
    - short texts (e.g., first 60 chars) of both segments.
- Manually listen to a couple of those timestamps in the audio to verify the logic makes sense.


----------------------------------------------
## 7. Turn-Taking Pattern & Alternation (feature 5.13)
----------------------------------------------

Goal:
Analyze the high-level conversation structure:
- how often speakers alternate
- how often one speaker keeps the floor
- produce simple metrics describing turn-taking dynamics.

[x] 7.1. Implement turn_taking_stats in analysis_speaking_features.py

Function:
- turn_taking_stats(
      transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
      output_path: str = "outputs/audio_features/turn_taking_stats.json"
  ) -> dict

Tasks:
- Load transcript_with_speakers.json.
- Sort segments by start time.
- Build a sequence of speakers in order (optionally merge immediately-adjacent segments of the same speaker with tiny gaps).
- Compute:
  - total_transitions = number of times we go from one segment to the next.
  - For each ordered speaker pair (A -> B):
    - count how many transitions occur where previous speaker is A and next speaker is B.
  - alternation_rate:
    - fraction of transitions where A != B.
- Also compute:
  - runs of the same speaker (consecutive segments with same speaker):
    - distribution of run lengths (in segments and in time).
    - longest_run_duration_sec per speaker.

Output JSON structure:
{
  "audio_file": "...",
  "speakers": ["Joe Rogan", "Donald Trump"],
  "transitions": {
    "Joe Rogan->Donald Trump": int,
    "Donald Trump->Joe Rogan": int,
    "Joe Rogan->Joe Rogan": int,
    "Donald Trump->Donald Trump": int
  },
  "total_transitions": int,
  "alternation_rate": float,   // fraction where speaker changes
  "runs": {
    "Joe Rogan": {
      "num_runs": int,
      "avg_run_segments": float,
      "avg_run_duration_sec": float,
      "max_run_duration_sec": float
    },
    "Donald Trump": {
      ...
    }
  }
}

Checks:
- Print a concise report to console:
    - transitions per pair (A->B)
    - alternation_rate (%)
    - max_run_duration per speaker.
- Optionally build a simple timeline visualization (not mandatory now):
    - e.g., print 1 char per segment: “A A A B B A …” or use a basic matplotlib colored bar plot later.


============================================================
END OF .PROMPT FILE
============================================================
