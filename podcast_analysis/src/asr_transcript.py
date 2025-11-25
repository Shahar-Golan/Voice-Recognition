"""
ASR (Automatic Speech Recognition) transcript generation module.

This module uses OpenAI Whisper to transcribe audio files and extract
timestamps for both segments and individual words.
"""

import os
import json
import whisper
import torch
import librosa
import numpy as np
from config import AUDIO_WAV_PATH

# Configure FFmpeg path for whisper
try:
    import imageio_ffmpeg as ffmpeg
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    # Set environment variable that whisper looks for
    os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
    print(f"✅ FFmpeg configured for whisper: {ffmpeg_path}")
except ImportError:
    print("⚠️ imageio-ffmpeg not available, hoping system FFmpeg works")


def transcribe_audio_with_whisper():
    """
    Transcribe audio file using OpenAI Whisper with timestamps.
    
    Inputs:
    - data/processed/podcast_16k_mono.wav
    
    Outputs:
    - outputs/audio_features/transcript_segments_for_audio.json
    
    Returns:
    - True if successful, False otherwise
    """
    # Get the project root directory (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create absolute paths
    audio_wav_path = os.path.join(project_root, "podcast_analysis", "data", "processed", "podcast_16k_mono.wav")
    output_json_path = os.path.join(project_root, "podcast_analysis", "outputs", "audio_features", "transcript_segments_for_audio.json")
    
    print("Starting ASR transcription with OpenAI Whisper...")
    print(f"Project root: {project_root}")
    print(f"Audio file: {audio_wav_path}")
    print(f"Output file: {output_json_path}")
    
    # Check if input file exists
    if not os.path.exists(audio_wav_path):
        print(f"❌ Error: Input audio file not found: {audio_wav_path}")
        print("Please ensure the audio preprocessing step has been completed.")
        return False
    
    # Check file size
    file_size = os.path.getsize(audio_wav_path) / (1024 * 1024)  # Size in MB
    print(f"Input file size: {file_size:.2f} MB")
    
    try:
        # Check if CUDA is available for faster processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load Whisper model (using 'base' for good balance of speed/accuracy)
        # Options: tiny, base, small, medium, large, large-v2, large-v3
        print("Loading Whisper model (base)...")
        model = whisper.load_model("base", device=device)
        
        # Load audio with librosa first (we know this works)
        print("Loading audio with librosa...")
        audio_data, sample_rate = librosa.load(audio_wav_path, sr=16000, mono=True)
        print(f"Audio loaded: {len(audio_data)} samples, {sample_rate} Hz")
        
        # Transcribe with word-level timestamps using pre-loaded audio
        print("Starting transcription (this may take a while for long audio)...")
        print("⏳ Processing... Please be patient.")
        
        result = model.transcribe(
            audio_data,           # Pass numpy array instead of file path
            word_timestamps=True,  # Enable word-level timestamps
            verbose=True          # Show progress
        )
        
        print(f"✅ Transcription completed!")
        print(f"Number of segments: {len(result['segments'])}")
        
        # Process and structure the results
        transcript_segments = []
        
        for segment in result['segments']:
            segment_data = {
                "start": segment['start'],
                "end": segment['end'], 
                "text": segment['text'].strip(),
                "words": []
            }
            
            # Add word-level timestamps if available
            if 'words' in segment and segment['words']:
                for word_info in segment['words']:
                    word_data = {
                        "start": word_info.get('start', segment['start']),
                        "end": word_info.get('end', segment['end']),
                        "word": word_info.get('word', '').strip()
                    }
                    segment_data['words'].append(word_data)
            
            transcript_segments.append(segment_data)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_json_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON file
        print(f"Saving transcript to: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_segments, f, indent=2, ensure_ascii=False)
        
        # Print summary statistics
        total_words = sum(len(seg['words']) for seg in transcript_segments)
        total_duration = transcript_segments[-1]['end'] if transcript_segments else 0
        
        print("\n📊 Transcription Summary:")
        print(f"  - Total segments: {len(transcript_segments)}")
        print(f"  - Total words: {total_words}")
        print(f"  - Audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"  - Average words per minute: {(total_words / (total_duration/60)):.1f}" if total_duration > 0 else "")
        
        # Show a few sample entries
        print(f"\n📝 Sample segments:")
        for i, segment in enumerate(transcript_segments[:3]):
            print(f"  Segment {i+1}: [{segment['start']:.2f}s - {segment['end']:.2f}s]")
            print(f"    Text: \"{segment['text'][:100]}{'...' if len(segment['text']) > 100 else ''}\"")
            print(f"    Words: {len(segment['words'])} words")
            if segment['words']:
                sample_words = segment['words'][:5]
                words_preview = ' '.join([w['word'] for w in sample_words])
                print(f"    First words: \"{words_preview}{'...' if len(segment['words']) > 5 else ''}\"")
            print()
        
        print(f"✅ Transcript saved to: {output_json_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_and_validate_transcript():
    """
    Load and validate the generated transcript file.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_json_path = os.path.join(project_root, "podcast_analysis", "outputs", "audio_features", "transcript_segments_for_audio.json")
    
    if not os.path.exists(output_json_path):
        print(f"❌ Transcript file not found: {output_json_path}")
        return None
    
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        print(f"✅ Loaded transcript with {len(transcript)} segments")
        return transcript
        
    except Exception as e:
        print(f"❌ Error loading transcript: {e}")
        return None


def main():
    """
    Main function to run ASR transcription.
    """
    print("🎤 Starting ASR Transcript Generation - Task 3.1")
    print("=" * 50)
    
    success = transcribe_audio_with_whisper()
    
    if success:
        print("\n🎉 ASR transcription completed successfully!")
        print("📁 Output file ready for audio feature extraction pipeline.")
        
        # Validate the output
        transcript = load_and_validate_transcript()
        if transcript:
            print("🔍 Transcript validation successful!")
    else:
        print("\n❌ ASR transcription failed. Please check the error messages above.")


if __name__ == "__main__":
    main()