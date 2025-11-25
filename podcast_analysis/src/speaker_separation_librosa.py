"""
Alternative speaker separation module using librosa instead of pydub.
This module splits stereo audio into separate speaker channels using librosa.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from config import SAMPLE_RATE


def split_stereo_to_speakers_librosa():
    """
    Split stereo podcast.mp3 into two mono speaker files using librosa.
    Left channel = Speaker A, Right channel = Speaker B.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get the project root directory (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create absolute paths with proper path separators
    audio_raw_path = os.path.join(project_root, "podcast_analysis", "data", "raw", "podcast.mp3")
    processed_dir = os.path.join(project_root, "podcast_analysis", "data", "processed")
    speaker_a_path = os.path.join(processed_dir, "podcast_speaker_A_16k_mono.wav")
    speaker_b_path = os.path.join(processed_dir, "podcast_speaker_B_16k_mono.wav")
    
    print("Starting speaker separation with librosa...")
    print(f"Project root: {project_root}")
    print(f"Looking for audio file: {audio_raw_path}")
    print(f"File exists: {os.path.exists(audio_raw_path)}")
    
    if os.path.exists(audio_raw_path):
        file_size = os.path.getsize(audio_raw_path) / (1024 * 1024)  # Size in MB
        print(f"Input file size: {file_size:.2f} MB")
    
    # Check if input file exists
    if not os.path.exists(audio_raw_path):
        print(f"Error: Input file {audio_raw_path} not found!")
        print("Please ensure the podcast.mp3 file is placed in the data/raw/ directory.")
        print(f"Expected location: {audio_raw_path}")
        return False
    
    try:
        # Load the MP3 file using librosa (preserving stereo)
        print("Loading audio with librosa...")
        audio_data, original_sr = librosa.load(audio_raw_path, sr=None, mono=False)
        
        # Check audio properties
        print(f"Original audio properties:")
        if audio_data.ndim == 1:
            print(f"  - Channels: 1 (mono)")
            channels = 1
        else:
            print(f"  - Channels: {audio_data.shape[0]}")
            channels = audio_data.shape[0]
        
        print(f"  - Sample rate: {original_sr} Hz")
        duration_sec = len(audio_data) / original_sr if audio_data.ndim == 1 else audio_data.shape[1] / original_sr
        print(f"  - Duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
        
        if channels != 2:
            print(f"⚠️ Warning: Audio is not stereo (has {channels} channels).")
            print("Speaker separation requires stereo audio with 2 channels.")
            print("If this is mono audio with 2 speakers, consider using speaker diarization instead.")
            return False
        
        print("✅ Audio is stereo - proceeding with speaker separation...")
        
        # Split stereo into two mono tracks
        print("Splitting stereo into left and right channels...")
        left_channel = audio_data[0]   # Speaker A (first channel)
        right_channel = audio_data[1]  # Speaker B (second channel)
        
        print("Processing channels...")
        print("  - Left channel (index 0) -> Speaker A")
        print("  - Right channel (index 1) -> Speaker B")
        
        # Resample to target sample rate if needed
        if original_sr != SAMPLE_RATE:
            print(f"Resampling from {original_sr} Hz to {SAMPLE_RATE} Hz...")
            left_channel = librosa.resample(left_channel, orig_sr=original_sr, target_sr=SAMPLE_RATE)
            right_channel = librosa.resample(right_channel, orig_sr=original_sr, target_sr=SAMPLE_RATE)
        else:
            print(f"Audio is already at target sample rate: {SAMPLE_RATE} Hz")
        
        # Create output directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        # Export Speaker A (left channel)
        print(f"Exporting Speaker A to: {speaker_a_path}")
        sf.write(speaker_a_path, left_channel, SAMPLE_RATE)
        
        # Export Speaker B (right channel)
        print(f"Exporting Speaker B to: {speaker_b_path}")
        sf.write(speaker_b_path, right_channel, SAMPLE_RATE)
        
        # Print final audio properties for confirmation
        print("\nFinal audio properties:")
        
        # Check Speaker A
        if os.path.exists(speaker_a_path):
            speaker_a_size = os.path.getsize(speaker_a_path) / (1024 * 1024)
            speaker_a_duration = len(left_channel) / SAMPLE_RATE
            print(f"Speaker A:")
            print(f"  - Channels: 1 (mono)")
            print(f"  - Sample rate: {SAMPLE_RATE} Hz")
            print(f"  - Duration: {speaker_a_duration:.2f} seconds ({speaker_a_duration/60:.2f} minutes)")
            print(f"  - File size: {speaker_a_size:.2f} MB")
            print(f"  - File: {speaker_a_path}")
        
        # Check Speaker B
        if os.path.exists(speaker_b_path):
            speaker_b_size = os.path.getsize(speaker_b_path) / (1024 * 1024)
            speaker_b_duration = len(right_channel) / SAMPLE_RATE
            print(f"Speaker B:")
            print(f"  - Channels: 1 (mono)")
            print(f"  - Sample rate: {SAMPLE_RATE} Hz")
            print(f"  - Duration: {speaker_b_duration:.2f} seconds ({speaker_b_duration/60:.2f} minutes)")
            print(f"  - File size: {speaker_b_size:.2f} MB")
            print(f"  - File: {speaker_b_path}")
        
        print("\n📝 Speaker mapping:")
        print("  - Speaker A = left channel (index 0)")
        print("  - Speaker B = right channel (index 1)")
        
        # Verify both output files were created
        if os.path.exists(speaker_a_path) and os.path.exists(speaker_b_path):
            print("✅ Speaker separation completed successfully!")
            return True
        else:
            print("❌ Error: One or both output files were not created!")
            return False
            
    except Exception as e:
        print(f"❌ Error during speaker separation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_speaker_file_paths():
    """
    Get the file paths for the separated speaker audio files.
    
    Returns:
        tuple: (speaker_a_path, speaker_b_path)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(project_root, "podcast_analysis", "data", "processed")
    speaker_a_path = os.path.join(processed_dir, "podcast_speaker_A_16k_mono.wav")
    speaker_b_path = os.path.join(processed_dir, "podcast_speaker_B_16k_mono.wav")
    return speaker_a_path, speaker_b_path


def verify_speaker_files():
    """
    Verify that speaker separation files exist and are valid.
    
    Returns:
        bool: True if both files exist and are valid, False otherwise
    """
    speaker_a_path, speaker_b_path = get_speaker_file_paths()
    
    print("Verifying speaker separation files...")
    
    # Check if files exist
    if not os.path.exists(speaker_a_path):
        print(f"❌ Speaker A file not found: {speaker_a_path}")
        return False
    
    if not os.path.exists(speaker_b_path):
        print(f"❌ Speaker B file not found: {speaker_b_path}")
        return False
    
    try:
        # Try to load files to verify they're valid audio
        audio_a, sr_a = librosa.load(speaker_a_path, sr=None)
        audio_b, sr_b = librosa.load(speaker_b_path, sr=None)
        
        print(f"✅ Speaker A: {len(audio_a)} samples at {sr_a} Hz")
        print(f"✅ Speaker B: {len(audio_b)} samples at {sr_b} Hz")
        
        if sr_a != SAMPLE_RATE or sr_b != SAMPLE_RATE:
            print(f"⚠️ Warning: Sample rates don't match expected {SAMPLE_RATE} Hz")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying speaker files: {e}")
        return False


def main():
    """
    Main function to run speaker separation using librosa.
    """
    print("🎤 Starting Speaker Separation with Librosa - Task 2.1")
    print("=" * 60)
    
    success = split_stereo_to_speakers_librosa()
    
    if success:
        print("\n🎉 Speaker separation completed successfully!")
        print("📁 Output files ready for individual speaker ASR transcription.")
        
        # Verify the output files
        if verify_speaker_files():
            print("🔍 Speaker file validation successful!")
        else:
            print("⚠️ Speaker file validation failed!")
    else:
        print("\n❌ Speaker separation failed. Please check the error messages above.")
        print("\n💡 Troubleshooting tips:")
        print("  1. Ensure podcast.mp3 is in data/raw/ directory")
        print("  2. Verify the audio file is stereo (2 channels)")
        print("  3. Check that you have sufficient disk space")
        print("  4. Ensure the audio file is not corrupted")


if __name__ == "__main__":
    main()