"""
Phase 1: JSON-Only Segmentation (Define Windows, No Audio Yet)
Goal: Use ONLY the JSON transcript to define "speech windows" for the target speaker.
No audio loading here.
"""

import json
import os
from typing import List, Dict, Any


# Configuration parameters
TARGET_SPEAKER = "Donald Trump"
MIN_KEEP_LEN = 1.0  # seconds - minimum length to keep before merging
MIN_WORDS = 3       # minimum number of words to keep
MIN_FINAL_LEN = 3.0  # seconds - minimum final window length
MAX_GAP = 1.0       # seconds - maximum gap allowed between merged pieces


def load_transcript(json_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON transcript and return sorted list of segments.
    
    Args:
        json_path: Path to transcript_with_speakers.json
        
    Returns:
        List of segment dictionaries, each containing:
        - speaker (str)
        - start (float, seconds)
        - end (float, seconds) 
        - text (str)
        - words (list of dicts with "start", "end", "word")
    """
    print(f"Loading transcript from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    print(f"Loaded {len(segments)} total segments")
    
    # Ensure segments are sorted by start time
    segments_sorted = sorted(segments, key=lambda x: x['start'])
    
    return segments_sorted


def filter_segments_by_speaker(segments: List[Dict[str, Any]], target_speaker: str) -> List[Dict[str, Any]]:
    """
    Filter segments to keep only those from the target speaker.
    
    Args:
        segments: List of all segments
        target_speaker: Name of target speaker to filter for
        
    Returns:
        List of segments from target speaker only
    """
    speaker_segments = [seg for seg in segments if seg.get('speaker') == target_speaker]
    print(f"Filtered to {len(speaker_segments)} segments for speaker: {target_speaker}")
    
    return speaker_segments


def drop_tiny_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove extremely tiny segments before merging.
    
    Args:
        segments: List of segments to filter
        
    Returns:
        List of segments with tiny ones removed
    """
    filtered_segments = []
    
    for seg in segments:
        duration = seg['end'] - seg['start']
        word_count = len(seg.get('words', []))
        
        # Keep if duration >= MIN_KEEP_LEN OR word_count >= MIN_WORDS
        if duration >= MIN_KEEP_LEN or word_count >= MIN_WORDS:
            filtered_segments.append(seg)
        else:
            print(f"Dropping tiny segment: {duration:.2f}s, {word_count} words - '{seg.get('text', '')[:50]}'")
    
    print(f"After dropping tiny segments: {len(filtered_segments)} remaining")
    return filtered_segments


def merge_adjacent_segments_for_speaker(segments: List[Dict[str, Any]], target_speaker: str) -> List[Dict[str, Any]]:
    """
    Merge adjacent segments into larger speech windows for a specific speaker.
    
    Logic:
    - Initialize empty list of windows
    - Iterate through segments in time order
    - If no current window, start one
    - If gap <= MAX_GAP, extend current window
    - Else close current window (if >= MIN_FINAL_LEN) and start new one
    - After loop, flush last window if >= MIN_FINAL_LEN
    
    Args:
        segments: List of segments to merge
        target_speaker: Name of the target speaker
        
    Returns:
        List of merged windows
    """
    if not segments:
        return []
    
    windows = []
    current_start = None
    current_end = None
    current_texts = []
    current_words = []
    
    for seg in segments:
        seg_start = seg['start']
        seg_end = seg['end']
        seg_text = seg.get('text', '')
        seg_words = seg.get('words', [])
        
        if current_start is None:
            # Start first window
            current_start = seg_start
            current_end = seg_end
            current_texts = [seg_text] if seg_text else []
            current_words = seg_words.copy()
        else:
            gap = seg_start - current_end
            
            if gap <= MAX_GAP:
                # Extend current window
                current_end = seg_end
                if seg_text:
                    current_texts.append(seg_text)
                current_words.extend(seg_words)
            else:
                # Close current window if it's long enough
                window_duration = current_end - current_start
                if window_duration >= MIN_FINAL_LEN:
                    window = {
                        'start': current_start,
                        'end': current_end,
                        'duration': window_duration,
                        'text': ' '.join(current_texts),
                        'word_count': len(current_words),
                        'speaker': target_speaker
                    }
                    windows.append(window)
                    print(f"Created window: {window_duration:.2f}s, {len(current_words)} words")
                
                # Start new window
                current_start = seg_start
                current_end = seg_end
                current_texts = [seg_text] if seg_text else []
                current_words = seg_words.copy()
    
    # Flush last window
    if current_start is not None:
        window_duration = current_end - current_start
        if window_duration >= MIN_FINAL_LEN:
            window = {
                'start': current_start,
                'end': current_end,
                'duration': window_duration,
                'text': ' '.join(current_texts),
                'word_count': len(current_words),
                'speaker': target_speaker
            }
            windows.append(window)
            print(f"Created final window: {window_duration:.2f}s, {len(current_words)} words")
    
    print(f"Created {len(windows)} final windows")
    return windows


def merge_adjacent_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent segments into larger speech windows.
    
    Logic:
    - Initialize empty list of windows
    - Iterate through segments in time order
    - If no current window, start one
    - If gap <= MAX_GAP, extend current window
    - Else close current window (if >= MIN_FINAL_LEN) and start new one
    - After loop, flush last window if >= MIN_FINAL_LEN
    
    Args:
        segments: List of segments to merge
        
    Returns:
        List of merged windows
    """
    if not segments:
        return []
    
    windows = []
    current_start = None
    current_end = None
    current_texts = []
    current_words = []
    
    for seg in segments:
        seg_start = seg['start']
        seg_end = seg['end']
        seg_text = seg.get('text', '')
        seg_words = seg.get('words', [])
        
        if current_start is None:
            # Start first window
            current_start = seg_start
            current_end = seg_end
            current_texts = [seg_text] if seg_text else []
            current_words = seg_words.copy()
        else:
            gap = seg_start - current_end
            
            if gap <= MAX_GAP:
                # Extend current window
                current_end = seg_end
                if seg_text:
                    current_texts.append(seg_text)
                current_words.extend(seg_words)
            else:
                # Close current window if it's long enough
                window_duration = current_end - current_start
                if window_duration >= MIN_FINAL_LEN:
                    window = {
                        'start': current_start,
                        'end': current_end,
                        'duration': window_duration,
                        'text': ' '.join(current_texts),
                        'word_count': len(current_words),
                        'speaker': TARGET_SPEAKER
                    }
                    windows.append(window)
                    print(f"Created window: {window_duration:.2f}s, {len(current_words)} words")
                
                # Start new window
                current_start = seg_start
                current_end = seg_end
                current_texts = [seg_text] if seg_text else []
                current_words = seg_words.copy()
    
    # Flush last window
    if current_start is not None:
        window_duration = current_end - current_start
        if window_duration >= MIN_FINAL_LEN:
            window = {
                'start': current_start,
                'end': current_end,
                'duration': window_duration,
                'text': ' '.join(current_texts),
                'word_count': len(current_words),
                'speaker': TARGET_SPEAKER
            }
            windows.append(window)
            print(f"Created final window: {window_duration:.2f}s, {len(current_words)} words")
    
    print(f"Created {len(windows)} final windows")
    return windows


def save_segmentation_metadata(windows: List[Dict[str, Any]], output_path: str, target_speaker: str = None) -> None:
    """
    Save segmentation metadata to disk.
    
    Args:
        windows: List of final windows
        output_path: Path to save JSON file
        target_speaker: Name of target speaker (defaults to global TARGET_SPEAKER)
    """
    if target_speaker is None:
        target_speaker = TARGET_SPEAKER
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add window IDs
    for i, window in enumerate(windows):
        window['window_id'] = f"seg_{i:04d}"
    
    # Create metadata structure
    metadata = {
        'target_speaker': target_speaker,
        'total_windows': len(windows),
        'total_duration': sum(w['duration'] for w in windows),
        'parameters': {
            'MIN_KEEP_LEN': MIN_KEEP_LEN,
            'MIN_WORDS': MIN_WORDS, 
            'MIN_FINAL_LEN': MIN_FINAL_LEN,
            'MAX_GAP': MAX_GAP
        },
        'windows': windows
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(windows)} windows to: {output_path}")
    
    # Print summary statistics
    durations = [w['duration'] for w in windows]
    word_counts = [w['word_count'] for w in windows]
    
    print("\nSummary Statistics:")
    print(f"Total windows: {len(windows)}")
    print(f"Total duration: {sum(durations):.1f} seconds")
    print(f"Average duration: {sum(durations)/len(durations):.1f} seconds")
    print(f"Min duration: {min(durations):.1f} seconds")
    print(f"Max duration: {max(durations):.1f} seconds")
    print(f"Average words per window: {sum(word_counts)/len(word_counts):.1f}")


def main():
    """Main function to run Phase 1 segmentation pipeline."""
    import sys
    
    # Check if speaker name is provided as argument
    if len(sys.argv) > 1:
        target_speaker = sys.argv[1]
    else:
        target_speaker = TARGET_SPEAKER
    
    print(f"=== PHASE 1: JSON-Only Segmentation for {target_speaker} ===")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "outputs", "audio_features", "transcript_with_speakers.json")
    
    # Create safe filename from speaker name
    safe_speaker_name = target_speaker.replace(" ", "_").replace(".", "")
    output_path = os.path.join(base_dir, "data", "segments", f"{safe_speaker_name}_segments.json")
    
    # Step 1.1: Load JSON transcript
    print("\nStep 1.1: Loading transcript...")
    segments = load_transcript(json_path)
    
    # Step 1.2: Filter by target speaker
    print(f"\nStep 1.2: Filtering for speaker '{target_speaker}'...")
    speaker_segments = filter_segments_by_speaker(segments, target_speaker)
    
    # Step 1.4: Drop extremely tiny segments (before merging)
    print(f"\nStep 1.4: Dropping tiny segments...")
    filtered_segments = drop_tiny_segments(speaker_segments)
    
    # Step 1.3: Merge adjacent segments
    print(f"\nStep 1.3: Merging adjacent segments...")
    windows = merge_adjacent_segments_for_speaker(filtered_segments, target_speaker)
    
    # Step 1.5: Save segmentation metadata
    print(f"\nStep 1.5: Saving metadata...")
    save_segmentation_metadata(windows, output_path, target_speaker)
    
    print(f"\n=== PHASE 1 COMPLETE for {target_speaker} ===")


if __name__ == "__main__":
    main()