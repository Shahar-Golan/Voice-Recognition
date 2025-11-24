"""
Alternative audio preprocessing using librosa.
This bypasses pydub/FFmpeg issues by using librosa directly.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from config import AUDIO_RAW_PATH, AUDIO_WAV_PATH, SAMPLE_RATE


def convert_to_mono_wav_librosa():
    """
    Convert MP3 podcast file to mono WAV format using librosa.
    """
    # Get the project root directory (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create absolute paths with proper path separators
    audio_raw_path = os.path.join(project_root, "podcast_analysis", "data", "raw", "podcast.mp3")
    audio_wav_path = os.path.join(project_root, "podcast_analysis", "data", "processed", "podcast_16k_mono.wav")
    
    print("Starting audio preprocessing with librosa...")
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
        # Load the MP3 file using librosa
        print("Loading audio with librosa...")
        audio_data, original_sr = librosa.load(audio_raw_path, sr=None, mono=False)
        
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
        
        # Convert to mono if needed
        if audio_data.ndim > 1:
            print("Converting to mono...")
            audio_data = librosa.to_mono(audio_data)
        
        # Resample to target sample rate if needed
        if original_sr != SAMPLE_RATE:
            print(f"Resampling from {original_sr} Hz to {SAMPLE_RATE} Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=SAMPLE_RATE)
        else:
            print(f"Audio is already at target sample rate: {SAMPLE_RATE} Hz")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(audio_wav_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as WAV using soundfile
        print(f"Exporting processed audio to: {audio_wav_path}")
        sf.write(audio_wav_path, audio_data, SAMPLE_RATE)
        
        # Print final audio properties for confirmation
        print("\nFinal audio properties:")
        print(f"  - Channels: 1 (mono)")
        print(f"  - Sample rate: {SAMPLE_RATE} Hz")
        final_duration = len(audio_data) / SAMPLE_RATE
        print(f"  - Duration: {final_duration:.2f} seconds ({final_duration/60:.2f} minutes)")
        print(f"  - Output file: {audio_wav_path}")
        
        # Verify the output file was created
        if os.path.exists(audio_wav_path):
            file_size = os.path.getsize(audio_wav_path) / (1024 * 1024)  # Size in MB
            print(f"  - File size: {file_size:.2f} MB")
            print("✅ Audio preprocessing completed successfully!")
            return True
        else:
            print("❌ Error: Output file was not created!")
            return False
            
    except Exception as e:
        print(f"❌ Error during audio preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to run audio preprocessing with librosa.
    """
    success = convert_to_mono_wav_librosa()
    if success:
        print("\n🎵 Audio is now ready for feature extraction!")
    else:
        print("\n❌ Audio preprocessing failed. Please check the error messages above.")


if __name__ == "__main__":
    main()