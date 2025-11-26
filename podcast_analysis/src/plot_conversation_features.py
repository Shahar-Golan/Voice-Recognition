"""
Plotting module for conversation features analysis.
Creates visualizations of speaker statistics, turn-taking patterns, interruptions, and other conversation dynamics.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define constants for project structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# Ensure plots directory exists at module import time
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Helper functions for loading data files
def load_basic_speaker_stats(path=None):
    """
    Load basic speaker statistics from JSON file.
    
    Args:
        path: Optional path to JSON file. Defaults to PROJECT_ROOT / "outputs/audio_features/basic_speaker_stats.json"
    
    Returns:
        pd.DataFrame: DataFrame with columns ["speaker", "total_speaking_time_sec", "total_words",
                     "num_segments", "total_speaking_time_min", "words_per_minute"]
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "audio_features" / "basic_speaker_stats.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert speakers dict to list of records
    records = []
    for speaker, stats in data['speakers'].items():
        record = {
            'speaker': speaker,
            'total_speaking_time_sec': stats['total_speaking_time_sec'],
            'total_words': stats['total_words'],
            'num_segments': stats['num_segments'],
            'total_speaking_time_min': stats['total_speaking_time_min'],
            'words_per_minute': stats['words_per_minute']
        }
        records.append(record)
    
    return pd.DataFrame(records)


def load_speaking_rate_timeseries(path=None):
    """
    Load speaking rate timeseries from JSON file.
    
    Args:
        path: Optional path to JSON file. Defaults to PROJECT_ROOT / "outputs/audio_features/speaking_rate_timeseries.json"
    
    Returns:
        pd.DataFrame: DataFrame with columns ["window_index", "window_start", "window_end",
                     "speaker", "word_count", "words_per_minute"]
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "audio_features" / "speaking_rate_timeseries.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert timeseries list to DataFrame
    timeseries_data = data['timeseries']
    df = pd.DataFrame(timeseries_data)
    
    return df


def load_turn_taking_stats(path=None):
    """
    Load turn-taking statistics from JSON file.
    
    Args:
        path: Optional path to JSON file. Defaults to PROJECT_ROOT / "outputs/audio_features/turn_taking_stats.json"
    
    Returns:
        tuple: (transitions_df, runs_df)
            - transitions_df: DataFrame with columns ["from_to", "count"]
            - runs_df: DataFrame with columns ["speaker", "num_runs", "avg_run_segments",
                      "avg_run_duration_sec", "max_run_duration_sec", "max_run_segments", 
                      "total_speaking_time_sec"]
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "audio_features" / "turn_taking_stats.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Create transitions DataFrame
    transitions_records = []
    for from_to, count in data['transitions'].items():
        transitions_records.append({'from_to': from_to, 'count': count})
    transitions_df = pd.DataFrame(transitions_records)
    
    # Create runs DataFrame
    runs_records = []
    for speaker, runs_stats in data['runs'].items():
        record = {
            'speaker': speaker,
            'num_runs': runs_stats['num_runs'],
            'avg_run_segments': runs_stats['avg_run_segments'],
            'avg_run_duration_sec': runs_stats['avg_run_duration_sec'],
            'max_run_duration_sec': runs_stats['max_run_duration_sec'],
            'max_run_segments': runs_stats['max_run_segments'],
            'total_speaking_time_sec': runs_stats['total_speaking_time_sec']
        }
        runs_records.append(record)
    runs_df = pd.DataFrame(runs_records)
    
    return transitions_df, runs_df


def load_interruptions(path=None):
    """
    Load interruption data from JSON file.
    
    Args:
        path: Optional path to JSON file. Defaults to PROJECT_ROOT / "outputs/audio_features/interruptions.json"
    
    Returns:
        tuple: (interruptions_df, per_speaker_df)
            - interruptions_df: DataFrame from "interruptions" list
            - per_speaker_df: DataFrame with columns ["speaker", "interruptions_made", 
                             "interruptions_received", "backchannels_made"]
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "audio_features" / "interruptions.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Create interruptions DataFrame from the list
    interruptions_df = pd.DataFrame(data['interruptions'])
    
    # Create per_speaker DataFrame from stats
    per_speaker_records = []
    for speaker, stats in data['stats']['per_speaker'].items():
        record = {
            'speaker': speaker,
            'interruptions_made': stats['interruptions_made'],
            'interruptions_received': stats['interruptions_received'],
            'backchannels_made': stats['backchannels_made']
        }
        per_speaker_records.append(record)
    per_speaker_df = pd.DataFrame(per_speaker_records)
    
    return interruptions_df, per_speaker_df


def load_diarization_segments(path=None):
    """
    Load diarization segments from JSON file.
    
    Args:
        path: Optional path to JSON file. Defaults to PROJECT_ROOT / "outputs/audio_features/diarization_segments.json"
    
    Returns:
        pd.DataFrame: DataFrame with columns ["speaker", "start", "end"]
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "audio_features" / "diarization_segments.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert segments list to DataFrame
    df = pd.DataFrame(data['segments'])
    
    return df


def load_transcript_with_speakers(path=None):
    """
    Load transcript with speakers from JSON file (optional for future use).
    
    Args:
        path: Optional path to JSON file. Defaults to PROJECT_ROOT / "outputs/audio_features/transcript_with_speakers.json"
    
    Returns:
        pd.DataFrame: DataFrame with transcript data
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "audio_features" / "transcript_with_speakers.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame - structure may vary depending on exact format
    # This is a placeholder implementation
    df = pd.DataFrame(data)
    
    return df


# TASK 2: CORE OVERVIEW PLOTS (GLOBAL CONVERSATION SHAPE)

def make_plot_total_speaking_time():
    """
    Create bar chart showing total speaking time in minutes per speaker.
    Saves as outputs/plots/01_total_speaking_time_by_speaker.png
    """
    df = load_basic_speaker_stats()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['speaker'], df['total_speaking_time_min'])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.title('Total Speaking Time (minutes) by Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Speaking Time (minutes)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "01_total_speaking_time_by_speaker.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_plot_total_words():
    """
    Create bar chart showing total words per speaker.
    Saves as outputs/plots/02_total_words_by_speaker.png
    """
    df = load_basic_speaker_stats()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['speaker'], df['total_words'])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.title('Total Words by Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Total Words', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "02_total_words_by_speaker.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_plot_speaking_rate_timeseries():
    """
    Create line plot showing speaking rate (words per minute) over time for each speaker.
    Saves as outputs/plots/03_speaking_rate_timeseries.png
    """
    df = load_speaking_rate_timeseries()
    
    # Compute window mid-point and convert to minutes
    df['window_mid'] = (df['window_start'] + df['window_end']) / 2.0
    df['window_mid_minutes'] = df['window_mid'] / 60.0
    
    plt.figure(figsize=(14, 8))
    
    # Plot line for each speaker
    speakers = df['speaker'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Default matplotlib colors
    
    for i, speaker in enumerate(speakers):
        speaker_data = df[df['speaker'] == speaker]
        plt.plot(speaker_data['window_mid_minutes'], 
                speaker_data['words_per_minute'], 
                label=speaker, 
                linewidth=2, 
                color=colors[i % len(colors)],
                marker='o', 
                markersize=3, 
                alpha=0.8)
    
    plt.title('Speaking Rate Over Time (Words per Minute, Sliding Window)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Words per Minute', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "03_speaking_rate_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# TASK 3: INTERRUPTIONS AND OVERLAPS PLOTS

def make_plot_interruptions_summary():
    """
    Create grouped bar chart showing interruptions and backchannels per speaker.
    Saves as outputs/plots/04_interruptions_summary_by_speaker.png
    """
    interruptions_df, per_speaker_df = load_interruptions()
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    speakers = per_speaker_df['speaker'].tolist()
    interruptions_made = per_speaker_df['interruptions_made'].tolist()
    interruptions_received = per_speaker_df['interruptions_received'].tolist()
    backchannels_made = per_speaker_df['backchannels_made'].tolist()
    
    # Set up bar positions
    x = range(len(speakers))
    width = 0.25
    
    # Create grouped bars
    plt.bar([i - width for i in x], interruptions_made, width, 
            label='Interruptions Made', color='#ff7f0e', alpha=0.8)
    plt.bar(x, interruptions_received, width, 
            label='Interruptions Received', color='#2ca02c', alpha=0.8)
    plt.bar([i + width for i in x], backchannels_made, width, 
            label='Backchannels Made', color='#d62728', alpha=0.8)
    
    # Add value labels on bars
    for i, (made, received, back) in enumerate(zip(interruptions_made, interruptions_received, backchannels_made)):
        plt.text(i - width, made + 5, str(made), ha='center', va='bottom', fontweight='bold')
        plt.text(i, received + 5, str(received), ha='center', va='bottom', fontweight='bold')
        plt.text(i + width, back + 2, str(back), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Interruptions and Backchannels per Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, speakers)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "04_interruptions_summary_by_speaker.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_plot_interruptions_timeline():
    """
    Create scatter plot showing interruptions over time and who interrupts whom.
    Saves as outputs/plots/05_interruptions_timeline.png
    """
    interruptions_df, per_speaker_df = load_interruptions()
    
    # Prepare data
    df = interruptions_df.copy()
    df['time_min'] = df['time'] / 60.0
    df['pair'] = df['interrupter'] + ' → ' + df['interrupted']
    
    plt.figure(figsize=(14, 8))
    
    # Get unique pairs and assign y-positions
    pairs = df['pair'].unique()
    pair_to_y = {pair: i for i, pair in enumerate(pairs)}
    df['y_pos'] = df['pair'].map(pair_to_y)
    
    # Create scatter plot with different colors for types
    types = df['type'].unique()
    colors = {'overlap': '#ff7f0e', 'quick_takeover': '#2ca02c'}
    
    for interrupt_type in types:
        type_data = df[df['type'] == interrupt_type]
        plt.scatter(type_data['time_min'], type_data['y_pos'], 
                   label=f'{interrupt_type.replace("_", " ").title()}',
                   color=colors.get(interrupt_type, '#1f77b4'),
                   alpha=0.6, s=50)
    
    plt.title('Interruptions Timeline: Who Interrupts Whom Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Interruption Direction', fontsize=12)
    plt.yticks(range(len(pairs)), pairs)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "05_interruptions_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_plot_interruption_duration_hist():
    """
    Create histogram showing distribution of interruption segment durations.
    Saves as outputs/plots/06_interruption_duration_hist.png
    """
    interruptions_df, per_speaker_df = load_interruptions()
    
    plt.figure(figsize=(12, 8))
    
    # Get duration data for each speaker
    trump_durations = interruptions_df[interruptions_df['interrupter'] == 'Donald Trump']['interrupter_duration']
    rogan_durations = interruptions_df[interruptions_df['interrupter'] == 'Joe Rogan']['interrupter_duration']
    
    # Create overlapping histograms
    plt.hist(trump_durations, bins=30, alpha=0.7, label='Donald Trump', 
             color='#ff7f0e', edgecolor='black', linewidth=0.5)
    plt.hist(rogan_durations, bins=30, alpha=0.7, label='Joe Rogan', 
             color='#2ca02c', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for mean durations
    trump_mean = trump_durations.mean()
    rogan_mean = rogan_durations.mean()
    plt.axvline(trump_mean, color='#ff7f0e', linestyle='--', linewidth=2, 
                label=f'Trump Mean: {trump_mean:.2f}s')
    plt.axvline(rogan_mean, color='#2ca02c', linestyle='--', linewidth=2, 
                label=f'Rogan Mean: {rogan_mean:.2f}s')
    
    plt.title('Distribution of Interruption Segment Durations', fontsize=14, fontweight='bold')
    plt.xlabel('Duration of Interrupter Segment (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "06_interruption_duration_hist.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_plot_interruption_types_by_speaker():
    """
    Create grouped bar chart comparing interruption types per speaker.
    Saves as outputs/plots/07_interruption_types_by_speaker.png
    """
    interruptions_df, per_speaker_df = load_interruptions()
    
    # Group by interrupter and type, count interruptions
    type_counts = interruptions_df.groupby(['interrupter', 'type']).size().reset_index(name='count')
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    speakers = type_counts['interrupter'].unique()
    types = type_counts['type'].unique()
    
    x = range(len(speakers))
    width = 0.35
    
    # Create bars for each type
    colors = {'overlap': '#ff7f0e', 'quick_takeover': '#2ca02c'}
    
    for i, interrupt_type in enumerate(types):
        type_data = type_counts[type_counts['type'] == interrupt_type]
        counts = []
        for speaker in speakers:
            speaker_data = type_data[type_data['interrupter'] == speaker]
            count = speaker_data['count'].values[0] if len(speaker_data) > 0 else 0
            counts.append(count)
        
        plt.bar([j + i * width for j in x], counts, width, 
                label=interrupt_type.replace('_', ' ').title(), 
                color=colors.get(interrupt_type, '#1f77b4'), 
                alpha=0.8)
        
        # Add value labels on bars
        for j, count in enumerate(counts):
            plt.text(j + i * width, count + 2, str(count), 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.title('Interruption Types per Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Interrupter', fontsize=12)
    plt.ylabel('Number of Interruptions', fontsize=12)
    plt.xticks([i + width/2 for i in x], speakers)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "07_interruption_types_by_speaker.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# TASK 5: OPTIONAL – SPEAKER TIME COURSE (STACKED AREA)

def make_plot_stacked_area_speaker_dominance():
    """
    Create stacked area chart showing proportion of speaking time per speaker over time.
    Saves as outputs/plots/11_stacked_area_speaker_dominance.png
    """
    segments_df = load_diarization_segments()
    
    # Determine time range and bin size
    max_time = segments_df['end'].max()
    bin_size = 60  # 1-minute bins
    num_bins = int(max_time / bin_size) + 1
    
    # Create time bins
    time_bins = [(i * bin_size, (i + 1) * bin_size) for i in range(num_bins)]
    
    # Get unique speakers
    speakers = segments_df['speaker'].unique()
    
    # Initialize data structure for results
    bin_data = []
    
    for i, (bin_start, bin_end) in enumerate(time_bins):
        bin_center = (bin_start + bin_end) / 2.0 / 60.0  # Convert to minutes
        speaker_times = {}
        
        for speaker in speakers:
            # Get segments for this speaker
            speaker_segments = segments_df[segments_df['speaker'] == speaker]
            total_speaking_time = 0
            
            for _, segment in speaker_segments.iterrows():
                seg_start, seg_end = segment['start'], segment['end']
                
                # Calculate overlap between segment and time bin
                overlap_start = max(seg_start, bin_start)
                overlap_end = min(seg_end, bin_end)
                
                if overlap_start < overlap_end:
                    total_speaking_time += overlap_end - overlap_start
            
            speaker_times[speaker] = total_speaking_time
        
        # Calculate total time and proportions
        total_time = sum(speaker_times.values())
        
        if total_time > 0:
            proportions = {speaker: time / total_time for speaker, time in speaker_times.items()}
        else:
            proportions = {speaker: 0 for speaker in speakers}
        
        bin_data.append({
            'time_min': bin_center,
            'total_time': total_time,
            **proportions
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(bin_data)
    
    # Create stacked area plot
    plt.figure(figsize=(16, 8))
    
    # Prepare data for stacking
    time_points = df['time_min']
    
    # Colors for speakers
    colors = {'Donald Trump': '#ff7f0e', 'Joe Rogan': '#2ca02c'}
    
    # Create stacked areas
    bottom = None
    for speaker in speakers:
        values = df[speaker]
        plt.fill_between(time_points, values if bottom is None else bottom, 
                        bottom + values if bottom is not None else values,
                        label=speaker, color=colors.get(speaker, '#1f77b4'), alpha=0.7)
        if bottom is None:
            bottom = values
        else:
            bottom = bottom + values
    
    # Add a line showing total speaking activity (how much of each minute has speech)
    activity = df['total_time'] / bin_size  # Proportion of each minute that has speech
    ax2 = plt.gca().twinx()
    ax2.plot(time_points, activity, 'k--', alpha=0.6, linewidth=1, 
             label='Speech Activity')
    ax2.set_ylabel('Speech Activity (Proportion of Time)', fontsize=11, alpha=0.7)
    ax2.set_ylim(0, 1)
    
    plt.title('Conversation Dominance Over Time\n(Proportion of Speaking Time per Minute)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Proportion of Speaking Time', fontsize=12)
    plt.xlim(0, max_time / 60)
    plt.ylim(0, 1)
    
    # Add legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "11_stacked_area_speaker_dominance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# TASK 4: TURN-TAKING AND CONVERSATION FLOW (we'll add this before Task 6)

def make_plot_transitions_bar():
    """
    Create bar chart showing speaker transition patterns (from->to).
    Saves as outputs/plots/08_transitions_bar.png
    """
    transitions_df, runs_df = load_turn_taking_stats()
    
    # Load the alternation rate from the original JSON for the title
    turn_taking_path = PROJECT_ROOT / "outputs" / "audio_features" / "turn_taking_stats.json"
    with open(turn_taking_path, 'r') as f:
        turn_data = json.load(f)
    alternation_rate = turn_data.get('alternation_rate', 0) * 100  # Convert to percentage
    
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(transitions_df['from_to'], transitions_df['count'], 
                   color=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
    
    # Add value labels on bars
    for bar, count in zip(bars, transitions_df['count']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Speaker Transitions (Alternation Rate: {alternation_rate:.1f}%)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Transition Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "08_transitions_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_plot_run_stats():
    """
    Create bar charts showing run statistics per speaker.
    Saves as outputs/plots/09_avg_run_duration_by_speaker.png and 10_max_run_duration_by_speaker.png
    """
    transitions_df, runs_df = load_turn_taking_stats()
    
    # Average run duration plot
    plt.figure(figsize=(10, 6))
    
    avg_durations_min = runs_df['avg_run_duration_sec'] / 60  # Convert to minutes
    bars = plt.bar(runs_df['speaker'], avg_durations_min)
    
    # Add value labels
    for bar, duration in zip(bars, avg_durations_min):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{duration:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Run Duration per Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Average Run Duration (minutes)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "09_avg_run_duration_by_speaker.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Maximum run duration plot
    plt.figure(figsize=(10, 6))
    
    max_durations_min = runs_df['max_run_duration_sec'] / 60  # Convert to minutes
    bars = plt.bar(runs_df['speaker'], max_durations_min)
    
    # Add value labels
    for bar, duration in zip(bars, max_durations_min):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{duration:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Maximum Run Duration per Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Maximum Run Duration (minutes)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "10_max_run_duration_by_speaker.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Creating conversation analysis plots...")
    
    try:
        # Task 2: Core Overview Plots
        print("\n=== Task 2: Core Overview Plots ===")
        print("Creating total speaking time plot...")
        make_plot_total_speaking_time()
        
        print("Creating total words plot...")
        make_plot_total_words()
        
        print("Creating speaking rate timeseries plot...")
        make_plot_speaking_rate_timeseries()
        
        # Task 3: Interruptions and Overlaps Plots
        print("\n=== Task 3: Interruptions and Overlaps Plots ===")
        print("Creating interruptions summary plot...")
        make_plot_interruptions_summary()
        
        print("Creating interruptions timeline plot...")
        make_plot_interruptions_timeline()
        
        print("Creating interruption duration histogram...")
        make_plot_interruption_duration_hist()
        
        print("Creating interruption types by speaker plot...")
        make_plot_interruption_types_by_speaker()
        
        # Task 4: Turn-Taking and Conversation Flow
        print("\n=== Task 4: Turn-Taking and Conversation Flow ===")
        print("Creating speaker transitions plot...")
        make_plot_transitions_bar()
        
        print("Creating run statistics plots...")
        make_plot_run_stats()
        
        # Task 5: Optional - Speaker Time Course (Stacked Area)
        print("\n=== Task 5: Speaker Time Course (Stacked Area) ===")
        print("Creating stacked area speaker dominance plot...")
        make_plot_stacked_area_speaker_dominance()
        
        print("\n" + "="*60)
        print("All plots completed successfully!")
        print("\nFiles saved in outputs/plots/:")
        print("Task 2 - Core Overview:")
        print("- 01_total_speaking_time_by_speaker.png")
        print("- 02_total_words_by_speaker.png") 
        print("- 03_speaking_rate_timeseries.png")
        print("\nTask 3 - Interruptions and Overlaps:")
        print("- 04_interruptions_summary_by_speaker.png")
        print("- 05_interruptions_timeline.png")
        print("- 06_interruption_duration_hist.png")
        print("- 07_interruption_types_by_speaker.png")
        print("\nTask 4 - Turn-Taking and Conversation Flow:")
        print("- 08_transitions_bar.png")
        print("- 09_avg_run_duration_by_speaker.png")
        print("- 10_max_run_duration_by_speaker.png")
        print("\nTask 5 - Speaker Time Course:")
        print("- 11_stacked_area_speaker_dominance.png")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        raise