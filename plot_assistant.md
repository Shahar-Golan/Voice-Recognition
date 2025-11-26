TASK 1: SETUP AND DATA LOADING

[ ] 1.1 Create plotting module and imports
- Create file: src/plot_conversation_features.py
- Add imports:
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

- Define a constant for the project root (optional) and plots directory:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

- At module import time, ensure PLOTS_DIR exists:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


[ ] 1.2 Implement loader helpers
- Implement helper functions:

  load_basic_speaker_stats(path=None) -> pd.DataFrame
    - Default path: PROJECT_ROOT / "basic_speaker_stats.json"
    - Load JSON.
    - Convert speakers dict into DataFrame with columns:
        ["speaker", "total_speaking_time_sec", "total_words",
         "num_segments", "total_speaking_time_min", "words_per_minute"]

  load_speaking_rate_timeseries(path=None) -> pd.DataFrame
    - Default path: PROJECT_ROOT / "speaking_rate_timeseries.json"
    - Load JSON.
    - Convert "timeseries" list into DataFrame with columns:
        ["window_index", "window_start", "window_end",
         "speaker", "word_count", "words_per_minute"]

  load_turn_taking_stats(path=None) -> (pd.DataFrame, pd.DataFrame)
    - Default path: PROJECT_ROOT / "turn_taking_stats.json"
    - Return:
      - transitions_df with columns ["from_to", "count"]
      - runs_df with columns:
          ["speaker", "num_runs", "avg_run_segments",
           "avg_run_duration_sec", "max_run_duration_sec",
           "max_run_segments", "total_speaking_time_sec"]
    - transitions_df:
        for each key "A->B" in transitions dict, create one row.

  load_interruptions(path=None) -> (pd.DataFrame, pd.DataFrame)
    - Default path: PROJECT_ROOT / "interruptions.json"
    - Return:
      - interruptions_df: DataFrame from "interruptions" list.
      - per_speaker_df: DataFrame from stats["per_speaker"], columns:
          ["speaker", "interruptions_made", "interruptions_received", "backchannels_made"]

  load_diarization_segments(path=None) -> pd.DataFrame
    - Default path: PROJECT_ROOT / "diarization_segments.json"
    - Columns: ["speaker", "start", "end"].

  (Optional) load_transcript_with_speakers(path=None) -> pd.DataFrame
    - Default path: PROJECT_ROOT / "transcript_with_speakers.json"
    - For now you don’t have to plot from this directly, but the function may be useful later.

TASK 2: CORE OVERVIEW PLOTS (GLOBAL CONVERSATION SHAPE)

[ ] 2.1 Plot total speaking time and words per speaker (bar chart)
Goal:
Show who dominates the conversation in total time and word count.

Input:
  - basic_speaker_stats.json via load_basic_speaker_stats().

Steps:
  - Use runs_df = load_basic_speaker_stats().
  - Create a bar chart with one bar per speaker for:
      (a) total_speaking_time_min
      (b) total_words
    You can:
      - either create two separate bar plots, or
      - use a subplot with two axes in a single figure.
  - X-axis: speaker names.
  - Y-axis: minutes (for speaking time plot) or #words (for word count plot).
  - Add exact values as text labels above bars if convenient.
  - Title examples:
      "Total Speaking Time (minutes) by Speaker"
      "Total Words by Speaker"
  - Save as:
      outputs/plots/01_total_speaking_time_by_speaker.png
      outputs/plots/02_total_words_by_speaker.png


[ ] 2.2 Time series of speaking rate per speaker (words per minute over time)
Goal:
Visualize how fast each speaker is talking over the interview, as a function of time.

Input:
  - speaking_rate_timeseries.json via load_speaking_rate_timeseries().

Steps:
  - Load df = load_speaking_rate_timeseries().
  - For each row, compute a "window_mid" column:
      window_mid = (window_start + window_end) / 2.0
  - Convert window_mid from seconds to minutes for display (optional but recommended).
  - Create a line plot:
      - X-axis: window_mid (minutes).
      - Y-axis: words_per_minute.
      - One line per speaker (filter df by speaker and plot on same axes).
  - Add legend mapping colors to speakers.
  - Add vertical grid lines lightly to show time progression.
  - Title example:
      "Speaking Rate Over Time (Words per Minute, Sliding Window)"
  - Save as:
      outputs/plots/03_speaking_rate_timeseries.png

TASK 3: INTERRUPTIONS AND OVERLAPS PLOTS

[ ] 3.1 Interruptions per speaker (summary bar chart)
Goal:
Show how many times each speaker interrupts the other, and how often they are interrupted.

Input:
  - interruptions.json via load_interruptions().

Steps:
  - Load interruptions_df, per_speaker_df = load_interruptions().
  - Create a grouped bar chart where:
    - X-axis: speaker ("Donald Trump", "Joe Rogan").
    - For each speaker, 2 (or 3) bars:
        - interruptions_made
        - interruptions_received
        (optionally) backchannels_made
  - Use a clear legend.
  - Title example:
      "Interruptions and Backchannels per Speaker"
  - Save as:
      outputs/plots/04_interruptions_summary_by_speaker.png


[ ] 3.2 Interruptions over time (timeline scatter plot)
Goal:
Show when interruptions occur across the conversation and who interrupts whom.

Input:
  - interruptions_df from load_interruptions().

Steps:
  - Use interruptions_df.
  - Create derived columns:
      - time_min = time / 60.0
      - pair = f"{interrupter} -> {interrupted}"
  - Option A (simple):
      - Scatter plot with:
          X-axis: time_min
          Y-axis: categorical values representing "Trump->Rogan" vs "Rogan->Trump"
            e.g., map pairs to y = 0 or 1.
      - Color points by type ("quick_takeover" vs "overlap").
  - Add legend for pair and/or type.
  - Title example:
      "Interruptions Timeline: Who Interrupts Whom Over Time"
  - Save as:
      outputs/plots/05_interruptions_timeline.png


[ ] 3.3 Distribution of interruption durations (histogram)
Goal:
Show how long interruptions last when someone takes the floor.

Input:
  - interruptions_df from load_interruptions().

Steps:
  - Use interrupter_duration column (in seconds).
  - Plot a histogram of interrupter_duration:
      - Optionally use different colors or overlays for each interrupter (two histograms).
  - X-axis: duration of interrupter segment (seconds).
  - Y-axis: count or density.
  - Add vertical line for mean duration per speaker if desired.
  - Title example:
      "Distribution of Interruption Segment Durations"
  - Save as:
      outputs/plots/06_interruption_duration_hist.png


[ ] 3.4 Interruption types by speaker (stacked or grouped bar)
Goal:
Compare how each speaker’s interruptions split between types (e.g. quick_takeover, overlap).

Input:
  - interruptions_df from load_interruptions().

Steps:
  - Group by ["interrupter", "type"] and count interruptions.
  - Create a grouped or stacked bar chart:
      - X-axis: interrupter.
      - Bars: count per type.
  - Title example:
      "Interruption Types per Speaker"
  - Save as:
      outputs/plots/07_interruption_types_by_speaker.png

TASK 4: TURN-TAKING AND CONVERSATION FLOW

[ ] 4.1 Speaker transition matrix (from->to)
Goal:
Visualize how often the conversation stays with the same speaker vs switches to the other.

Input:
  - transitions_df from load_turn_taking_stats().

Steps:
  - transitions_df has:
      columns: ["from_to", "count"], where "from_to" is "A->B".
  - Option A (simple bar):
      - Bar chart with:
          X-axis: from_to strings.
          Y-axis: count.
  - Make sure to show both:
      - same-speaker transitions: Trump->Trump, Rogan->Rogan
      - cross-speaker transitions: Trump->Rogan, Rogan->Trump
  - Add the alternation_rate (from turn_taking_stats.json) in the title or as text annotation.
  - Title example:
      "Speaker Transitions (Alternation Rate: XX%)"
  - Save as:
      outputs/plots/08_transitions_bar.png


[ ] 4.2 Run statistics per speaker (how long they keep the floor)
Goal:
Show how long each speaker tends to speak in a row before yielding the floor.

Input:
  - runs_df from load_turn_taking_stats().

Steps:
  - runs_df has:
      columns:
        ["speaker", "num_runs", "avg_run_segments",
         "avg_run_duration_sec", "max_run_duration_sec",
         "max_run_segments", "total_speaking_time_sec"]
  - Create a bar chart for avg_run_duration_sec per speaker:
      - X-axis: speaker.
      - Y-axis: avg_run_duration_sec (convert to minutes if desired).
  - Optionally create a second bar chart for max_run_duration_sec.
  - Titles examples:
      "Average Run Duration per Speaker (Seconds)"
      "Maximum Run Duration per Speaker (Seconds)"
  - Save as:
      outputs/plots/09_avg_run_duration_by_speaker.png
      outputs/plots/10_max_run_duration_by_speaker.png

TASK 5: OPTIONAL – SPEAKER TIME COURSE (STACKED AREA)

[ ] 5.1 Stacked area: who is speaking over time (from diarization)
Goal:
Give a high-level picture of how much of each minute is dominated by each speaker.

Input:
  - diarization_segments.json via load_diarization_segments().

Steps (optional, more advanced):
  - Load df = load_diarization_segments().
  - Bucket time into 1-minute bins (or 30-second bins).
  - For each bin and each speaker, compute total speaking time within that bin.
  - Compute proportion of speaking time per speaker per bin.
  - Create a stacked area chart:
      - X-axis: time (minutes, bin center).
      - Y-axis: proportion of speaking time (0–1).
      - Each area = one speaker.
  - Title example:
      "Conversation Dominance Over Time (Proportion of Speaking Time per Minute)"
  - Save as:
      outputs/plots/11_stacked_area_speaker_dominance.png

TASK 6: MAIN ENTRYPOINT

[ ] 6.1 Implement main() in src/plot_conversation_features.py
- Define:

    def main():
        # 1) Ensure plots directory exists (already done via PLOTS_DIR)
        # 2) Call each plot-making function in logical order:
        #    - make_plot_total_speaking_time()
        #    - make_plot_total_words()
        #    - make_plot_speaking_rate_timeseries()
        #    - make_plot_interruptions_summary()
        #    - make_plot_interruptions_timeline()
        #    - make_plot_interruption_duration_hist()
        #    - make_plot_interruption_types_by_speaker()
        #    - make_plot_transitions_bar()
        #    - make_plot_run_stats()
        #    - (optional) make_plot_stacked_area_speaker_dominance()
        # 3) Print a small summary: which files were saved.

    if __name__ == "__main__":
        main()

- Make sure the script runs from project root:
    (.venv) > python src/plot_conversation_features.py

- After running, verify that all .png files exist under:
    outputs/plots/


END OF PLOT_PROMPT FILE