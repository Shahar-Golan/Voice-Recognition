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
PHASE 1 – JSON-ONLY Segmentation (Define Windows, No Audio Yet)  **PHASE 1 COMPLETED ✅**
-------------------------------------------------------------------------------
Goal: Use ONLY the JSON transcript to define "speech windows" for the target speaker.
No audio loading here.

[✅] Create script: `src/build_segments_from_json.py`

[✅] Step 1.1 – Load JSON transcript
    - Implement function:
      `load_transcript(outputs\audio_features\transcript_with_speakers.json) -> list_of_segments`
      Each segment: dict with keys:
        - speaker (str)
        - start (float, seconds)
        - end   (float, seconds)
        - text  (str, optional)
        - words (list of dicts with "start", "end", "word", optional)
    - Ensure the result is sorted by "start" time.

[✅] Step 1.2 – Filter by target speaker "Donald Trump"
    - Decide:
        TARGET_SPEAKER = "Donald Trump"
    - Implement:
        `filter_segments_by_speaker(segments, target) -> speaker_segments`
      Keep only segments where:
        `segment["speaker"] == TARGET_SPEAKER`

[✅] Step 1.3 – Merge adjacent segments into larger speech windows
    Idea:
      - We want windows representing coherent speech chunks by the same speaker.
      - Use small gaps to merge segments.

[✅] Step 1.4 – Drop extremely tiny segments
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

[✅] Step 1.5 – Save segmentation metadata to disk
    - Save as `data/segments/Donald_Trump_segments.json`:
        - start_time (float, seconds)
        - end_time   (float, seconds)
        - duration   (end_time - start_time)
        - speaker    (always the target speaker)
        - text       (the same way it is in outputs\audio_features\transcript_with_speakers.json)



**Summary:**
- Successfully implemented JSON-only segmentation pipeline in `src/build_segments_from_json.py`
- Generated 503 speech windows for Donald Trump (119.5 min) and 184 for Joe Rogan (41.6 min) 
- Created comprehensive speaker comparison visualizations showing Trump spoke 74.2% of total time

-------------------------------------------------------------------------------
PHASE 2 – Use Metadata + Mono Audio to Cut WAV Files  **PHASE 2 COMPLETED ✅**
-------------------------------------------------------------------------------
Goal: Use the M windows from Phase 1 + the mono WAV to create M segment WAV files.

[✅] Create script: `src/slice_audio_segments.py`

[✅] Step 2.1 – Load mono audio
    - Use librosa to load:
        `data/processed/podcast_16k_mono.wav`
      E.g.:
        y, sr = librosa.load("data/processed/podcast_16k_mono.wav", sr=None)

[✅] Step 2.2 – Load segmentation metadata
    - Read `data/segments/[Speaker]_segments.json` into memory.
    - For each segment:
        - segment_id
        - start_time
        - end_time

[✅] Step 2.3 – Slice audio per segment
    - Implement:
        `slice_audio(y, sr, start_t, end_t) -> y_segment`
      with:
        i1 = int(start_t * sr)
        i2 = int(end_t * sr)
        return y[i1:i2]

[✅] Step 2.4 – Save each segment as a WAV file on disk
    - For each segment in metadata:
        - Compute y_segment = slice_audio(...)
        - Save to:
            `data/segments/audio/[Speaker]/seg_[id].wav`
          e.g.:
            `data/segments/audio/Donald_Trump/seg_001.wav`
    - Keep the same sample rate `sr`.

[✅] Step 2.5 – Verify consistency
    - Number of WAV files in `data/segments/audio/[Speaker]/` should equal number of segments in metadata.
    - Durations (in samples) should roughly match (end_time - start_time).

**Summary:**
- Successfully implemented audio slicing pipeline in `src/slice_audio_segments.py`
- Generated 502 WAV segments for Donald Trump and 184 for Joe Rogan (total: 686 segments)
- Directory structure: `data/segments/audio/[Speaker]/seg_[id].wav`
- All segments verified for duration accuracy with perfect match between metadata and actual audio

-------------------------------------------------------------------------------
PHASE 3 – Feature Extraction Per Segment WAV
-------------------------------------------------------------------------------
Goal: For each segment WAV file, compute prosodic features, including:
      1) pitch variability
      2) amplitude dynamics
      3) formant dispersion
      4) pitch change rate

[ ] Create script: `src/extract_features.py`

[ ] Step 3.1 – Iterate over segment WAV files and metadata
    - For each row in `segments_metadata.csv`:
        - Load the corresponding WAV file: `seg_XXXX.wav`
        - Compute basic features:

          (a) Duration
              - duration_sec = len(y_seg) / sr

          (b) Pitch contour and pitch-based features
              - Extract F0 contour per frame inside the segment:
                    F0 = [f1, f2, ..., fT]
                (e.g., using librosa/praats-parselmouth)

              - pitch_mean       = mean(F0)
              - pitch_std        = std(F0)
              - pitch_range      = max(F0) − min(F0)

              - pitch_variability = pitch_std
                (optionally you may also log pitch_range)

              - pitch_change_rate:
                    Δf_t = |f_t − f_(t−1)|  for t = 2..T
                    pitch_change_rate = (1 / (T−1)) * Σ_t Δf_t

          (c) Amplitude / energy features
              - Compute RMS energy per frame: RMS = [r1, r2, ..., rT]

              - rms_mean  = mean(RMS)
              - rms_std   = std(RMS)
              - rms_range = max(RMS) − min(RMS)

              - amplitude_dynamics = rms_std
                (optionally also store rms_range)

          (d) Formant features (using e.g. Parselmouth/Praat)
              - Estimate formants F1, F2, F3 per frame:
                    F1[t], F2[t], F3[t]

              - For each frame t:
                    D12[t] = F2[t] − F1[t]
                    D23[t] = F3[t] − F2[t]

              - formant_dispersion_12_mean = mean(D12)
              - formant_dispersion_23_mean = mean(D23)

              - formant_dispersion = (formant_dispersion_12_mean
                                      + formant_dispersion_23_mean) / 2
                (or keep both D12 and D23 as separate columns)

          (e) Speaking rate (optional, using JSON info)
              - words_per_second = (number_of_words_in_window) / duration_sec

          (f) Pauses ratio (optional)
              - Estimate voiced vs unvoiced/silent frames
              - pauses_ratio = (time_in_silence) / duration_sec

[ ] Step 3.2 – Save raw feature table
    - Create `data/features/segments_features_raw.csv` with columns, e.g.:

        - segment_id
        - start_time, end_time, duration_sec

        # Pitch-related
        - pitch_mean
        - pitch_std
        - pitch_range
        - pitch_variability       # usually = pitch_std
        - pitch_change_rate

        # Amplitude-related
        - rms_mean
        - rms_std
        - rms_range
        - amplitude_dynamics      # usually = rms_std

        # Formant-related
        - formant_dispersion_12_mean
        - formant_dispersion_23_mean
        - formant_dispersion

        # Optional extras
        - speaking_rate           # words_per_second
        - pauses_ratio

        # (any other features you add later)

-------------------------------------------------------------------------------
PHASE 4 – Normalize Features & Compute Arousal Index
-------------------------------------------------------------------------------
Goal: Convert raw features into standardized scores and compute a single
      Arousal Index based primarily on your prosodic features.

[ ] Create script: `src/compute_arousal.py`

[ ] Step 4.1 – Load raw features
    - Read `data/features/segments_features_raw.csv` into pandas.

[ ] Step 4.2 – Compute z-scores per feature
    For each numeric feature X that you want to use in arousal:

        z(X) = (X − μ_X) / σ_X

    - Compute μ_X and σ_X across all segments:
        μ_X = mean(X)
        σ_X = std(X)

    - Create new columns, for example:
        - z_pitch_variability
        - z_amplitude_dynamics
        - z_formant_dispersion
        - z_pitch_change_rate
        - z_pauses_ratio          (if you use pauses in the index)

[ ] Step 4.3 – Define Arousal Index per segment

    One reasonable initial definition:

        ArousalIndex = z_pitch_variability
                     + z_amplitude_dynamics
                     + z_pitch_change_rate
                     − z_pauses_ratio

    In formula form:

        z(X) = (X − μ_X) / σ_X

        ArousalIndex = z(pitch_variability)
                     + z(amplitude_dynamics)
                     + z(pitch_change_rate)
                     − z(pauses_ratio)

    - formant_dispersion can be kept as an additional descriptor:
        *z_formant_dispersion is stored and can be analyzed separately
        or added later to ArousalIndex once you experiment with it.*

[ ] Step 4.4 – Save with Arousal Index
    - Save to `data/features/segments_features_with_arousal.csv`
      with all original columns plus:
        - z_pitch_variability
        - z_amplitude_dynamics
        - z_formant_dispersion
        - z_pitch_change_rate
        - z_pauses_ratio
        - arousal_index

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
