"""
Analysis of speaking features from conversation transcripts.

This module provides functions to analyze conversation dynamics including:
- Basic speaker statistics
- Speaking rate time series
- Interruption detection
- Turn-taking patterns

Constants for interruption analysis:
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# Constants for interruption analysis
MIN_OVERLAP_SEC = 0.2
MAX_GAP_SEC = 0.15
MIN_WORDS_INTERRUPTER = 3
MAX_BACKCHANNEL_DURATION_SEC = 0.6


def basic_speaker_stats(
    transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
    output_path: str = "outputs/audio_features/basic_speaker_stats.json"
) -> dict:
    """
    Calculate basic statistics per speaker including total speaking time,
    words per minute, and number of segments.
    
    Args:
        transcript_path: Path to the transcript with speakers JSON file
        output_path: Path where to save the statistics JSON file
        
    Returns:
        Dictionary containing speaker statistics
    """
    # Load transcript data
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize speaker statistics
    speaker_stats = {}
    
    # Process each segment
    for segment in data['segments']:
        speaker = segment['speaker']
        duration = segment['end'] - segment['start']
        num_words = len(segment.get('words', []))
        
        # Initialize speaker data if not exists
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'total_speaking_time_sec': 0.0,
                'total_words': 0,
                'num_segments': 0
            }
        
        # Aggregate data
        speaker_stats[speaker]['total_speaking_time_sec'] += duration
        speaker_stats[speaker]['total_words'] += num_words
        speaker_stats[speaker]['num_segments'] += 1
    
    # Calculate derived statistics
    for speaker, stats in speaker_stats.items():
        stats['total_speaking_time_min'] = stats['total_speaking_time_sec'] / 60.0
        if stats['total_speaking_time_min'] > 0:
            stats['words_per_minute'] = stats['total_words'] / stats['total_speaking_time_min']
        else:
            stats['words_per_minute'] = 0.0
    
    # Create output structure
    result = {
        'audio_file': data['audio_file'],
        'speakers': speaker_stats
    }
    
    # Print summary table to console
    print("\n" + "="*70)
    print("BASIC SPEAKER STATISTICS")
    print("="*70)
    print(f"{'Speaker':<15} {'Time(min)':<10} {'Words':<8} {'WPM':<6} {'Segments':<8}")
    print("-" * 70)
    
    total_time = 0.0
    total_words = 0
    total_segments = 0
    
    for speaker, stats in speaker_stats.items():
        print(f"{speaker:<15} {stats['total_speaking_time_min']:<10.1f} "
              f"{stats['total_words']:<8} {stats['words_per_minute']:<6.0f} "
              f"{stats['num_segments']:<8}")
        
        total_time += stats['total_speaking_time_min']
        total_words += stats['total_words']
        total_segments += stats['num_segments']
    
    print("-" * 70)
    print(f"{'TOTAL':<15} {total_time:<10.1f} {total_words:<8} "
          f"{total_words/total_time if total_time > 0 else 0:<6.0f} {total_segments:<8}")
    print("="*70)
    
    # Save to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved basic speaker statistics to: {output_path}")
    
    return result


def speaking_rate_timeseries(
    transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
    output_data_path: str = "outputs/audio_features/speaking_rate_timeseries.json",
    output_plot_path: str = "outputs/audio_features/speaking_rate_timeseries.png",
    window_size_sec: float = 30.0,
    step_size_sec: Optional[float] = None
) -> dict:
    """
    Compute a time series of speaking rate (word count over time) per speaker.
    
    Args:
        transcript_path: Path to the transcript with speakers JSON file
        output_data_path: Path where to save the time-series data JSON file
        output_plot_path: Path where to save the plot PNG file
        window_size_sec: Size of time windows in seconds
        step_size_sec: Step size for sliding windows (None = non-overlapping)
        
    Returns:
        Dictionary containing time-series data
    """
    # Load transcript data
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine conversation bounds
    all_segments = data['segments']
    conversation_start = min(segment['start'] for segment in all_segments)
    conversation_end = max(segment['end'] for segment in all_segments)
    
    print(f"\nConversation duration: {conversation_start:.1f}s to {conversation_end:.1f}s "
          f"({(conversation_end - conversation_start)/60:.1f} minutes)")
    
    # Set step size (non-overlapping by default)
    if step_size_sec is None:
        step_size_sec = window_size_sec
    
    # Get all unique speakers
    speakers = list(set(segment['speaker'] for segment in all_segments))
    speakers.sort()  # For consistent ordering
    
    print(f"Speakers: {speakers}")
    print(f"Window size: {window_size_sec}s, Step size: {step_size_sec}s")
    
    # Generate time windows
    timeseries_data = []
    window_index = 0
    current_time = conversation_start
    
    while current_time < conversation_end:
        window_start = current_time
        window_end = min(current_time + window_size_sec, conversation_end)
        
        # Count words for each speaker in this window
        for speaker in speakers:
            word_count = 0
            
            # Check all segments for words in this window
            for segment in all_segments:
                if segment['speaker'] == speaker:
                    # Count words whose start time falls within the window
                    for word in segment.get('words', []):
                        word_start = word['start']
                        if window_start <= word_start < window_end:
                            word_count += 1
            
            # Calculate words per minute
            window_duration_min = (window_end - window_start) / 60.0
            words_per_minute = (word_count / window_duration_min) if window_duration_min > 0 else 0.0
            
            timeseries_data.append({
                "window_index": window_index,
                "window_start": window_start,
                "window_end": window_end,
                "speaker": speaker,
                "word_count": word_count,
                "words_per_minute": words_per_minute
            })
        
        window_index += 1
        current_time += step_size_sec
    
    # Create output structure
    result = {
        "audio_file": data['audio_file'],
        "window_size_sec": window_size_sec,
        "step_size_sec": step_size_sec,
        "timeseries": timeseries_data
    }
    
    # Print basic info
    num_windows = len(set(entry['window_index'] for entry in timeseries_data))
    print(f"\nGenerated {num_windows} time windows")
    print(f"Total time-series entries: {len(timeseries_data)}")
    
    # Show sample entries
    print("\nSample time-series entries:")
    for i, entry in enumerate(timeseries_data[:6]):  # First 3 windows (2 speakers each)
        print(f"  Window {entry['window_index']}: {entry['speaker']} - "
              f"{entry['word_count']} words, {entry['words_per_minute']:.1f} WPM "
              f"({entry['window_start']:.1f}-{entry['window_end']:.1f}s)")
    
    # Verify word count consistency (optional check)
    total_words_timeseries = sum(entry['word_count'] for entry in timeseries_data)
    print(f"\nWord count verification:")
    print(f"  Total words in time-series: {total_words_timeseries}")
    
    # Save data to JSON
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved time-series data to: {output_data_path}")
    
    # Create plot
    _create_timeseries_plot(timeseries_data, speakers, output_plot_path)
    
    return result


def _create_timeseries_plot(timeseries_data: List[Dict], speakers: List[str], output_plot_path: str):
    """Create and save a matplotlib plot of the speaking rate time series."""
    
    # Organize data by speaker
    speaker_data = {speaker: {'times': [], 'wpm': []} for speaker in speakers}
    
    for entry in timeseries_data:
        speaker = entry['speaker']
        # Use window center time for x-axis
        window_center = (entry['window_start'] + entry['window_end']) / 2
        speaker_data[speaker]['times'].append(window_center / 60.0)  # Convert to minutes
        speaker_data[speaker]['wpm'].append(entry['words_per_minute'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, speaker in enumerate(speakers):
        color = colors[i % len(colors)]
        plt.plot(speaker_data[speaker]['times'], speaker_data[speaker]['wpm'], 
                label=speaker, color=color, linewidth=2, marker='o', markersize=3)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Words per Minute')
    plt.title('Speaking Rate Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved time-series plot to: {output_plot_path}")


if __name__ == "__main__":
    # Test the functions
    print("Testing analysis_speaking_features functions...")
    
    # Use relative path from the src directory
    transcript_path = "../outputs/audio_features/transcript_with_speakers.json"
    
    try:
        # Test basic_speaker_stats function
        print("\n" + "="*70)
        print("TESTING BASIC SPEAKER STATS")
        print("="*70)
        basic_output_path = "../outputs/audio_features/basic_speaker_stats.json"
        stats = basic_speaker_stats(transcript_path, basic_output_path)
        print("✓ Basic speaker stats completed successfully!")
        
        # Test speaking_rate_timeseries function
        print("\n" + "="*70)
        print("TESTING SPEAKING RATE TIME SERIES")
        print("="*70)
        timeseries_data_path = "../outputs/audio_features/speaking_rate_timeseries.json"
        timeseries_plot_path = "../outputs/audio_features/speaking_rate_timeseries.png"
        
        timeseries = speaking_rate_timeseries(
            transcript_path=transcript_path,
            output_data_path=timeseries_data_path,
            output_plot_path=timeseries_plot_path,
            window_size_sec=30.0,
            step_size_sec=None  # Non-overlapping windows
        )
        print("✓ Speaking rate time-series completed successfully!")
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()