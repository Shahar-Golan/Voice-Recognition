Title: Prosodic Windows + Heated Segments Pipeline (JSON-first segmentation)

Goal:
Build a pipeline that:
1) Uses ONLY the JSON transcript to define "speech windows" for a single target speaker.
2) Cleans/merges those windows (remove too-short segments, merge adjacent ones, etc.).
3) Saves metadata for M final segments (start/end/duration/text) to disk.
4) Uses that metadata + mono audio to cut and save M WAV files.
5) Extracts prosodic features from each WAV.
6) Computes an Arousal Index and marks each segment as "heated" or not.

-------------------------------------------------------------------------------
PHASE 1 – JSON-ONLY Segmentation (Define Windows, No Audio Yet)
-------------------------------------------------------------------------------
Goal: Use ONLY the JSON transcript to define "speech windows" for the target speaker.
No audio loading here.

[ ] Create script: `src/build_segments_from_json.py`

[ ] Step 1.1 – Load JSON transcript
    - Implement function:
      `load_transcript(outputs\audio_features\transcript_with_speakers.json) -> list_of_segments`
      Each segment: dict with keys:
        - speaker (str)
        - start (float, seconds)
        - end   (float, seconds)
        - text  (str, optional)
        - words (list of dicts with "start", "end", "word", optional)
    - Ensure the result is sorted by "start" time.

[ ] Step 1.2 – Filter by target speaker "Donald Trump"
    - Decide:
        TARGET_SPEAKER = "Donald Trump"
    - Implement:
        `filter_segments_by_speaker(segments, target) -> speaker_segments`
      Keep only segments where:
        `segment["speaker"] == TARGET_SPEAKER`

[ ] Step 1.3 – Merge adjacent segments into larger speech windows
    Idea:
      - We want windows representing coherent speech chunks by the same speaker.
      - Use small gaps to merge segments.

[ ] Step 1.4 – Drop extremely tiny segments
    - Define:
        MIN_KEEP_LEN = up to 1.0 second OR up to 3 words 
    - Remove segments with:
        (end - start) < MIN_KEEP_LEN
    - This is just to reduce noise before merging.

    Parameters:
      - MIN_FINAL_LEN = 3.0 seconds   (below that we don't keep)
      - MIN_WORDS = 3
      - MAX_GAP = 1.0 seconds         (max silence allowed between merged pieces)

    Logic (JSON-only):
      - Initialize empty list `windows = []`
      - Iterate through outputs\audio_features\transcript_with_speakers.json in time order:
          * If there is no current window:
               current_start = seg.start
               current_end   = seg.end
            else:
               gap    = seg.start - current_end
               new_len = seg.end - current_start
               If (gap <= MAX_GAP) AND (new_len <= MAX_FINAL_LEN):
                    → extend current window:
                       current_end = seg.end
               Else:
                    → close current window:
                       if (current_end - current_start) >= MIN_FINAL_LEN:
                           append window (current_start, current_end, optional text)
                       start a new window at this segment:
                           current_start = seg.start
                           current_end   = seg.end
      - After loop:
          * flush last window if duration ≥ MIN_FINAL_LEN

    Result of Phase 1:
      - List of M final windows:
          window_id
          start_time
          end_time
          duration
          (optional) concatenated text from JSON segments inside the window

[ ] Step 1.5 – Save segmentation metadata to disk
    - Save as `data/segments/Donald_Trump_segments.json`:
        - start_time (float, seconds)
        - end_time   (float, seconds)
        - duration   (end_time - start_time)
        - speaker    (always the target speaker)
        - text       (the same way it is in outputs\audio_features\transcript_with_speakers.json)

-------------------------------------------------------------------------------
PHASE 2 – Use Metadata + Mono Audio to Cut WAV Files
-------------------------------------------------------------------------------
Goal: Use the M windows from Phase 1 + the mono WAV to create M segment WAV files.

[ ] Create script: `src/slice_audio_segments.py`

[ ] Step 2.1 – Load mono audio
    - Use librosa (or soundfile) to load:
        `data/raw/podcast_mono.wav`
      E.g.:
        y, sr = librosa.load("data/raw/podcast_mono.wav", sr=None)

[ ] Step 2.2 – Load segmentation metadata
    - Read `data/segments/segments_metadata.csv` into memory.
    - For each row:
        - segment_id
        - start_time
        - end_time

[ ] Step 2.3 – Slice audio per segment
    - Implement:
        `slice_audio(y, sr, start_t, end_t) -> y_segment`
      with:
        i1 = int(start_t * sr)
        i2 = int(end_t * sr)
        return y[i1:i2]

[ ] Step 2.4 – Save each segment as a WAV file on disk
    - For each row in metadata:
        - Compute y_segment = slice_audio(...)
        - Save to:
            `data/segments/<segment_id>.wav`
          e.g.:
            `data/segments/seg_0001.wav`
    - Keep the same sample rate `sr`.

[ ] Step 2.5 – Verify consistency
    - Number of WAV files in `data/segments/` should equal number of rows in metadata.
    - Durations (in samples) should roughly match (end_time - start_time).

-------------------------------------------------------------------------------
PHASE 3 – Feature Extraction Per Segment WAV
-------------------------------------------------------------------------------
Goal: For each segment WAV file, compute prosodic features.

[ ] Create script: `src/extract_features.py`

[ ] Step 3.1 – Iterate over segment WAV files and metadata
    - For each row in `segments_metadata.csv`:
        - Load the corresponding WAV file: `seg_XXXX.wav`
        - Compute basic features:
            - duration (seconds) = len(y_seg) / sr
            - pitch statistics:
                * mean F0
                * std F0
            - RMS energy:
                * mean RMS
                * std RMS
            - speaking rate (optional, using JSON info):
                * words_per_second = (number_of_words_in_window) / duration
            - pauses ratio (optional, later):
                * proportion of silence vs voiced samples

[ ] Step 3.2 – Save raw feature table
    - Create `data/features/segments_features_raw.csv` with columns:
        - segment_id
        - start_time, end_time, duration
        - pitch_mean, pitch_std
        - rms_mean, rms_std
        - speaking_rate
        - pauses_ratio
        - (any other features you add later)

-------------------------------------------------------------------------------
PHASE 4 – Normalize Features & Compute Arousal Index
-------------------------------------------------------------------------------
Goal: Convert raw features into standardized scores and compute a single Arousal Index.

[ ] Create script: `src/compute_arousal.py`

[ ] Step 4.1 – Load raw features
    - Read `data/features/segments_features_raw.csv` into pandas.

[ ] Step 4.2 – Compute z-scores per feature
    For each numeric feature X used in arousal:
        z(X) = (X − μ_X) / σ_X
    - Compute μ_X and σ_X across all segments:
        μ_X = mean(X)
        σ_X = std(X)
    - Create new columns:
        z_pitch_mean
        z_rms_mean
        z_speaking_rate
        z_pauses_ratio

[ ] Step 4.3 – Define Arousal Index per segment
    ArousalIndex = z_pitch_mean + z_rms_mean + z_speaking_rate − z_pauses_ratio

    In formula form:

        z(X) = (X - μ_X) / σ_X

        ArousalIndex = z(pitch_mean)
                     + z(rms_mean)
                     + z(speaking_rate)
                     − z(pauses_ratio)

[ ] Step 4.4 – Save with Arousal Index
    - Save to `data/features/segments_features_with_arousal.csv`

-------------------------------------------------------------------------------
PHASE 5 – Label Heated vs Non-Heated Segments
-------------------------------------------------------------------------------
Goal: Decide which segments are "heated" based on Arousal Index.

[ ] Step 5.1 – Compute threshold
    - Load `segments_features_with_arousal.csv`
    - Compute:
        μ_A = mean(arousal_index)
        σ_A = std(arousal_index)
    - Choose k (e.g., k = 1.5 or 2.0)

[ ] Step 5.2 – Define heated label
    Heated(segment) = 1 if arousal_index ≥ μ_A + k · σ_A
                      0 otherwise

[ ] Step 5.3 – Add `is_heated` column
    - Add a column in the DataFrame:
        `is_heated` = 1/0 (or True/False)

[ ] Step 5.4 – Save final feature table
    - Save as:
        `data/features/segments_final.csv`
      with:
        - segment_id
        - start_time, end_time, duration
        - all features
        - arousal_index
        - is_heated

-------------------------------------------------------------------------------
PHASE 6 – Optional: Visualization & Further Work
-------------------------------------------------------------------------------
[ ] Plot arousal_index over time (segment start_time or midpoint).
[ ] Highlight segments with is_heated == 1.
[ ] Listen to some heated vs non-heated segments to validate subjectively.
[ ] Later:
    - Align segments with semantic topics.
    - Cluster segments by prosodic style.
    - Train ML models using these features and labels.

-------------------------------------------------------------------------------
END OF .PROMPT FILE
-------------------------------------------------------------------------------
