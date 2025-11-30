"""
Phase 2: Use Metadata + Mono Audio to Cut WAV Files
Goal: Use the M windows from Phase 1 + the mono WAV to create M segment WAV files.
"""

import json
import os
import sys
import librosa
import soundfile as sf
import numpy as np
from typing import Dict, Any, List


def load_mono_audio(audio_path: str) -> tuple:
    """
    Load mono audio file using librosa.
    
    Args:
        audio_path: Path to the mono WAV file
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    print(f"Loading audio from: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio with original sample rate
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"Loaded audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.1f} seconds)")
    return y, sr


def load_segmentation_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """
    Load segmentation metadata from JSON file.
    
    Args:
        metadata_path: Path to speaker segments JSON file
        
    Returns:
        List of window dictionaries
    """
    print(f"Loading metadata from: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    windows = data.get('windows', [])
    target_speaker = data.get('target_speaker', 'Unknown')
    
    print(f"Loaded {len(windows)} windows for speaker: {target_speaker}")
    return windows, target_speaker


def slice_audio(y: np.ndarray, sr: int, start_t: float, end_t: float) -> np.ndarray:
    """
    Slice audio segment from the full audio array.
    
    Args:
        y: Full audio array
        sr: Sample rate
        start_t: Start time in seconds
        end_t: End time in seconds
        
    Returns:
        Sliced audio segment
    """
    i1 = int(start_t * sr)
    i2 = int(end_t * sr)
    
    # Ensure indices are within bounds
    i1 = max(0, i1)
    i2 = min(len(y), i2)
    
    if i1 >= i2:
        print(f"Warning: Invalid time range {start_t:.2f}-{end_t:.2f}s, returning empty segment")
        return np.array([])
    
    return y[i1:i2]


def save_segment_wav(y_segment: np.ndarray, sr: int, output_path: str) -> None:
    """
    Save audio segment as WAV file.
    
    Args:
        y_segment: Audio segment data
        sr: Sample rate
        output_path: Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as WAV file
    sf.write(output_path, y_segment, sr)


def verify_segments(output_dir: str, windows: List[Dict[str, Any]]) -> None:
    """
    Verify that all segments were created successfully.
    
    Args:
        output_dir: Directory containing WAV files
        windows: List of window metadata
    """
    print("\n=== VERIFICATION ===")
    
    # Count WAV files
    wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    wav_count = len(wav_files)
    expected_count = len(windows)
    
    print(f"Expected segments: {expected_count}")
    print(f"Created WAV files: {wav_count}")
    
    if wav_count == expected_count:
        print("✅ All segments created successfully!")
    else:
        print(f"❌ Mismatch: Expected {expected_count}, got {wav_count}")
    
    # Check a few segments for duration consistency
    print("\nSample duration checks:")
    for i, window in enumerate(windows[:5]):  # Check first 5
        window_id = window['window_id']
        expected_duration = window['duration']
        wav_path = os.path.join(output_dir, f"{window_id}.wav")
        
        if os.path.exists(wav_path):
            # Load and check duration
            y_check, sr_check = librosa.load(wav_path, sr=None)
            actual_duration = len(y_check) / sr_check
            
            diff = abs(actual_duration - expected_duration)
            status = "✅" if diff < 0.1 else "⚠️"
            
            print(f"  {window_id}: Expected {expected_duration:.2f}s, Got {actual_duration:.2f}s {status}")
        else:
            print(f"  {window_id}: File missing ❌")


def main():
    """Main function to run Phase 2 audio slicing pipeline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Slice audio segments for a speaker')
    parser.add_argument('speaker', help='Speaker name (e.g., "Joe Rogan", "Donald Trump")')
    args = parser.parse_args()
    
    target_speaker = args.speaker
    print(f"=== PHASE 2: Audio Slicing for {target_speaker} ===")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input paths
    audio_path = os.path.join(base_dir, "data", "processed", "podcast_16k_mono.wav")
    safe_speaker_name = target_speaker.replace(" ", "_").replace(".", "")
    metadata_path = os.path.join(base_dir, "data", "segments", f"{safe_speaker_name}_segments.json")
    
    # Output directory - now includes 'audio' subdirectory
    output_dir = os.path.join(base_dir, "data", "segments", "audio", safe_speaker_name)
    
    try:
        # Step 2.1: Load mono audio
        print("\nStep 2.1: Loading mono audio...")
        y, sr = load_mono_audio(audio_path)
        
        # Step 2.2: Load segmentation metadata
        print("\nStep 2.2: Loading segmentation metadata...")
        windows, loaded_speaker = load_segmentation_metadata(metadata_path)
        
        if loaded_speaker.lower() != target_speaker.lower():
            print(f"Warning: Requested speaker '{target_speaker}' doesn't match metadata speaker '{loaded_speaker}'")
        
        # Step 2.3 & 2.4: Slice and save audio segments
        print(f"\nStep 2.3-2.4: Slicing and saving {len(windows)} segments...")
        
        successful_segments = 0
        failed_segments = 0
        
        for i, window in enumerate(windows):
            window_id = window['window_id']
            start_time = window['start']
            end_time = window['end']
            expected_duration = window['duration']
            
            # Step 2.3: Slice audio
            y_segment = slice_audio(y, sr, start_time, end_time)
            
            if len(y_segment) == 0:
                print(f"Failed to slice segment {window_id}")
                failed_segments += 1
                continue
            
            # Step 2.4: Save segment as WAV file
            output_path = os.path.join(output_dir, f"{window_id}.wav")
            save_segment_wav(y_segment, sr, output_path)
            
            successful_segments += 1
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(windows)} segments...")
        
        print(f"\nSegment processing complete!")
        print(f"  Successful: {successful_segments}")
        print(f"  Failed: {failed_segments}")
        print(f"  Output directory: {output_dir}")
        
        # Step 2.5: Verify consistency
        print("\nStep 2.5: Verifying consistency...")
        verify_segments(output_dir, windows)
        
        print(f"\n=== PHASE 2 COMPLETE for {target_speaker} ===")
        
    except Exception as e:
        print(f"Error during audio slicing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()