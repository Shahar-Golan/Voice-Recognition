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


def detect_interruptions(
    transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
    output_path: str = "outputs/audio_features/interruptions.json",
    min_overlap_sec: float = 0.2,
    max_gap_sec: float = 0.15,
    min_words_interrupter: int = 3,
    max_backchannel_duration_sec: float = 0.6
) -> dict:
    """
    Detect meaningful interruptions between speakers by analyzing overlaps and gaps.
    
    Args:
        transcript_path: Path to the transcript with speakers JSON file
        output_path: Path where to save the interruptions JSON file
        min_overlap_sec: Minimum overlap duration to consider interruption
        max_gap_sec: Maximum gap for "instant takeover" interruptions
        min_words_interrupter: Minimum words needed to be real interruption
        max_backchannel_duration_sec: Max duration for backchannel classification
        
    Returns:
        Dictionary containing interruption analysis results
    """
    # Load transcript data
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sort segments by start time
    segments = sorted(data['segments'], key=lambda x: x['start'])
    
    print(f"\nAnalyzing {len(segments)} segments for interruptions...")
    print(f"Parameters:")
    print(f"  Min overlap: {min_overlap_sec}s")
    print(f"  Max gap: {max_gap_sec}s") 
    print(f"  Min words for interruption: {min_words_interrupter}")
    print(f"  Max backchannel duration: {max_backchannel_duration_sec}s")
    
    interruptions = []
    backchannels = []
    
    # Analyze consecutive segment pairs
    for i in range(len(segments) - 1):
        current_segment = segments[i]
        next_segment = segments[i + 1]
        
        # Skip if same speaker (no interruption possible)
        if current_segment['speaker'] == next_segment['speaker']:
            continue
            
        current_start = current_segment['start']
        current_end = current_segment['end']
        next_start = next_segment['start']
        next_end = next_segment['end']
        
        # Calculate overlap and gap
        overlap_start = max(current_start, next_start)
        overlap_end = min(current_end, next_end)
        overlap_duration = max(0.0, overlap_end - overlap_start)
        gap = next_start - current_end
        
        interruption_detected = False
        interruption_type = None
        
        # Case 1: Overlap (hard interruption)
        if overlap_duration >= min_overlap_sec:
            interruption_detected = True
            interruption_type = "overlap"
            
        # Case 2: Near-zero gap (soft interruption) 
        elif 0 <= gap <= max_gap_sec:
            interruption_detected = True
            interruption_type = "quick_takeover"
        
        if interruption_detected:
            # Determine who interrupted whom
            if interruption_type == "overlap":
                # The interrupter is the one who started speaking later during overlap
                if next_start > current_start:
                    interrupter = next_segment['speaker']
                    interrupted = current_segment['speaker']
                    interrupter_segment = next_segment
                    interrupted_segment = current_segment
                    interrupter_index = i + 1
                    interrupted_index = i
                else:
                    interrupter = current_segment['speaker']
                    interrupted = next_segment['speaker']
                    interrupter_segment = current_segment
                    interrupted_segment = next_segment
                    interrupter_index = i
                    interrupted_index = i + 1
            else:  # quick_takeover
                interrupter = next_segment['speaker']
                interrupted = current_segment['speaker']
                interrupter_segment = next_segment
                interrupted_segment = current_segment
                interrupter_index = i + 1
                interrupted_index = i
            
            # Count words in the interrupter segment
            num_words = len(interrupter_segment.get('words', []))
            segment_duration = interrupter_segment['end'] - interrupter_segment['start']
            
            # Classify as backchannel or real interruption
            if (num_words < min_words_interrupter and 
                segment_duration < max_backchannel_duration_sec):
                # This is likely a backchannel
                backchannels.append({
                    "time": interrupter_segment['start'],
                    "speaker": interrupter,
                    "segment_index": interrupter_index,
                    "text": interrupter_segment['text'],
                    "duration": segment_duration,
                    "word_count": num_words
                })
            else:
                # This is a real interruption
                interruptions.append({
                    "time": interrupter_segment['start'],
                    "overlap_duration": overlap_duration,
                    "gap": gap,
                    "interrupter": interrupter,
                    "interrupted": interrupted,
                    "interrupter_segment_index": interrupter_index,
                    "interrupted_segment_index": interrupted_index,
                    "interrupter_segment_text": interrupter_segment['text'][:100] + ("..." if len(interrupter_segment['text']) > 100 else ""),
                    "interrupted_segment_text": interrupted_segment['text'][:100] + ("..." if len(interrupted_segment['text']) > 100 else ""),
                    "type": interruption_type,
                    "interrupter_word_count": num_words,
                    "interrupter_duration": segment_duration
                })
    
    # Calculate statistics
    speakers = list(set(segment['speaker'] for segment in segments))
    per_speaker_stats = {}
    
    for speaker in speakers:
        per_speaker_stats[speaker] = {
            "interruptions_made": sum(1 for intr in interruptions if intr['interrupter'] == speaker),
            "interruptions_received": sum(1 for intr in interruptions if intr['interrupted'] == speaker),
            "backchannels_made": sum(1 for bc in backchannels if bc['speaker'] == speaker)
        }
    
    # Create output structure
    result = {
        "audio_file": data['audio_file'],
        "parameters": {
            "min_overlap_sec": min_overlap_sec,
            "max_gap_sec": max_gap_sec,
            "min_words_interrupter": min_words_interrupter,
            "max_backchannel_duration_sec": max_backchannel_duration_sec
        },
        "interruptions": interruptions,
        "backchannels": backchannels,
        "stats": {
            "total_interruptions": len(interruptions),
            "total_backchannels": len(backchannels),
            "per_speaker": per_speaker_stats
        }
    }
    
    # Print summary
    print(f"\n" + "="*70)
    print("INTERRUPTION ANALYSIS RESULTS")
    print("="*70)
    print(f"Total interruptions detected: {len(interruptions)}")
    print(f"Total backchannels detected: {len(backchannels)}")
    print(f"\nPer-speaker statistics:")
    print(f"{'Speaker':<15} {'Made':<8} {'Received':<10} {'Backchannels':<12}")
    print("-" * 50)
    
    for speaker, stats in per_speaker_stats.items():
        print(f"{speaker:<15} {stats['interruptions_made']:<8} "
              f"{stats['interruptions_received']:<10} {stats['backchannels_made']:<12}")
    
    # Show sample interruptions
    if interruptions:
        print(f"\nSample interruptions (first 5):")
        for i, intr in enumerate(interruptions[:5]):
            print(f"  {i+1}. {intr['time']/60:.1f}min: {intr['interrupter']} interrupted {intr['interrupted']}")
            print(f"     Type: {intr['type']}, Overlap: {intr['overlap_duration']:.2f}s, Gap: {intr['gap']:.2f}s")
            print(f"     Interrupter: \"{intr['interrupter_segment_text']}\"")
            print(f"     Interrupted: \"{intr['interrupted_segment_text']}\"")
            print()
    
    # Show sample backchannels
    if backchannels:
        print(f"Sample backchannels (first 3):")
        for i, bc in enumerate(backchannels[:3]):
            print(f"  {i+1}. {bc['time']/60:.1f}min: {bc['speaker']} - \"{bc['text'][:50]}...\"")
    
    print("="*70)
    
    # Save to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved interruption analysis to: {output_path}")
    
    return result


def turn_taking_stats(
    transcript_path: str = "outputs/audio_features/transcript_with_speakers.json",
    output_path: str = "outputs/audio_features/turn_taking_stats.json"
) -> dict:
    """
    Analyze turn-taking patterns and alternation in the conversation.
    
    Args:
        transcript_path: Path to the transcript with speakers JSON file
        output_path: Path where to save the turn-taking stats JSON file
        
    Returns:
        Dictionary containing turn-taking analysis results
    """
    # Load transcript data
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sort segments by start time
    segments = sorted(data['segments'], key=lambda x: x['start'])
    
    print(f"\nAnalyzing turn-taking patterns for {len(segments)} segments...")
    
    # Build speaker sequence (optionally merge adjacent same-speaker segments with tiny gaps)
    speaker_sequence = []
    merged_segments = []
    
    for i, segment in enumerate(segments):
        speaker = segment['speaker']
        start_time = segment['start']
        end_time = segment['end']
        
        # Check if this should be merged with previous segment
        if (merged_segments and 
            merged_segments[-1]['speaker'] == speaker and
            start_time - merged_segments[-1]['end'] <= 0.5):  # Small gap threshold
            # Merge with previous segment
            merged_segments[-1]['end'] = end_time
            merged_segments[-1]['segment_count'] += 1
            merged_segments[-1]['total_duration'] = merged_segments[-1]['end'] - merged_segments[-1]['start']
        else:
            # Start new merged segment
            merged_segments.append({
                'speaker': speaker,
                'start': start_time,
                'end': end_time,
                'segment_count': 1,
                'total_duration': end_time - start_time
            })
    
    # Build speaker sequence from merged segments
    speaker_sequence = [seg['speaker'] for seg in merged_segments]
    
    print(f"Merged {len(segments)} original segments into {len(merged_segments)} speaking turns")
    
    # Get unique speakers
    speakers = list(set(speaker_sequence))
    speakers.sort()  # For consistent ordering
    
    # Calculate transitions
    transitions = {}
    for speaker_a in speakers:
        for speaker_b in speakers:
            transitions[f"{speaker_a}->{speaker_b}"] = 0
    
    total_transitions = 0
    alternations = 0  # Transitions where speaker changes
    
    # Count transitions
    for i in range(len(speaker_sequence) - 1):
        current_speaker = speaker_sequence[i]
        next_speaker = speaker_sequence[i + 1]
        
        transition_key = f"{current_speaker}->{next_speaker}"
        transitions[transition_key] += 1
        total_transitions += 1
        
        if current_speaker != next_speaker:
            alternations += 1
    
    # Calculate alternation rate
    alternation_rate = alternations / total_transitions if total_transitions > 0 else 0.0
    
    # Analyze runs (consecutive segments with same speaker)
    runs_data = {}
    
    for speaker in speakers:
        runs_data[speaker] = {
            'run_segments': [],  # List of run lengths in segments
            'run_durations': [],  # List of run durations in seconds
            'num_runs': 0
        }
    
    # Find runs
    current_run_speaker = speaker_sequence[0] if speaker_sequence else None
    current_run_length = 1
    current_run_start_idx = 0
    
    for i in range(1, len(speaker_sequence)):
        if speaker_sequence[i] == current_run_speaker:
            current_run_length += 1
        else:
            # End of current run
            if current_run_speaker:
                # Calculate run duration
                run_start_time = merged_segments[current_run_start_idx]['start']
                run_end_time = merged_segments[i - 1]['end']
                run_duration = run_end_time - run_start_time
                
                runs_data[current_run_speaker]['run_segments'].append(current_run_length)
                runs_data[current_run_speaker]['run_durations'].append(run_duration)
                runs_data[current_run_speaker]['num_runs'] += 1
            
            # Start new run
            current_run_speaker = speaker_sequence[i]
            current_run_length = 1
            current_run_start_idx = i
    
    # Don't forget the last run
    if current_run_speaker and len(speaker_sequence) > 0:
        run_start_time = merged_segments[current_run_start_idx]['start']
        run_end_time = merged_segments[-1]['end']
        run_duration = run_end_time - run_start_time
        
        runs_data[current_run_speaker]['run_segments'].append(current_run_length)
        runs_data[current_run_speaker]['run_durations'].append(run_duration)
        runs_data[current_run_speaker]['num_runs'] += 1
    
    # Calculate run statistics
    runs_stats = {}
    for speaker in speakers:
        run_segments = runs_data[speaker]['run_segments']
        run_durations = runs_data[speaker]['run_durations']
        
        if run_segments:
            runs_stats[speaker] = {
                'num_runs': len(run_segments),
                'avg_run_segments': sum(run_segments) / len(run_segments),
                'avg_run_duration_sec': sum(run_durations) / len(run_durations),
                'max_run_duration_sec': max(run_durations),
                'max_run_segments': max(run_segments),
                'total_speaking_time_sec': sum(run_durations)
            }
        else:
            runs_stats[speaker] = {
                'num_runs': 0,
                'avg_run_segments': 0.0,
                'avg_run_duration_sec': 0.0,
                'max_run_duration_sec': 0.0,
                'max_run_segments': 0,
                'total_speaking_time_sec': 0.0
            }
    
    # Create output structure
    result = {
        "audio_file": data['audio_file'],
        "speakers": speakers,
        "transitions": transitions,
        "total_transitions": total_transitions,
        "alternation_rate": alternation_rate,
        "runs": runs_stats,
        "merged_segments_info": {
            "original_segments": len(segments),
            "merged_segments": len(merged_segments),
            "merge_gap_threshold_sec": 0.5
        }
    }
    
    # Print analysis results
    print(f"\n" + "="*70)
    print("TURN-TAKING ANALYSIS RESULTS")
    print("="*70)
    print(f"Total transitions: {total_transitions}")
    print(f"Alternations (speaker changes): {alternations}")
    print(f"Alternation rate: {alternation_rate:.2%}")
    
    print(f"\nTransition Matrix:")
    print(f"{'Transition':<25} {'Count':<8} {'%':<8}")
    print("-" * 42)
    for transition, count in transitions.items():
        percentage = (count / total_transitions * 100) if total_transitions > 0 else 0
        print(f"{transition:<25} {count:<8} {percentage:<8.1f}")
    
    print(f"\nSpeaking Run Statistics:")
    print(f"{'Speaker':<15} {'Runs':<6} {'Avg Segments':<12} {'Avg Duration(s)':<15} {'Max Duration(s)':<15}")
    print("-" * 70)
    for speaker in speakers:
        stats = runs_stats[speaker]
        print(f"{speaker:<15} {stats['num_runs']:<6} "
              f"{stats['avg_run_segments']:<12.1f} "
              f"{stats['avg_run_duration_sec']:<15.1f} "
              f"{stats['max_run_duration_sec']:<15.1f}")
    
    # Show example speaker sequence (first 20 turns)
    print(f"\nFirst 20 speaking turns:")
    sequence_preview = ' -> '.join(speaker_sequence[:20])
    if len(speaker_sequence) > 20:
        sequence_preview += " -> ..."
    print(f"  {sequence_preview}")
    
    print("="*70)
    
    # Save to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved turn-taking analysis to: {output_path}")
    
    return result


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
        
        # Test detect_interruptions function
        print("\n" + "="*70)
        print("TESTING INTERRUPTION DETECTION")
        print("="*70)
        interruptions_output_path = "../outputs/audio_features/interruptions.json"
        
        interruptions = detect_interruptions(
            transcript_path=transcript_path,
            output_path=interruptions_output_path,
            min_overlap_sec=0.2,
            max_gap_sec=0.15,
            min_words_interrupter=3,
            max_backchannel_duration_sec=0.6
        )
        print("✓ Interruption detection completed successfully!")
        
        # Test turn_taking_stats function
        print("\n" + "="*70)
        print("TESTING TURN-TAKING ANALYSIS")
        print("="*70)
        turn_taking_output_path = "../outputs/audio_features/turn_taking_stats.json"
        
        turn_taking = turn_taking_stats(
            transcript_path=transcript_path,
            output_path=turn_taking_output_path
        )
        print("✓ Turn-taking analysis completed successfully!")
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()