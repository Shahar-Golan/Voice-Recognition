"""
Speaker Diarization Module

This module implements speaker diarization for podcast audio using pyannote.audio.
It processes mono audio files and identifies when different speakers are talking.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def diarize_podcast(
    audio_file_path: str = "data/processed/podcast_16k_mono.wav",
    output_file_path: str = "outputs/audio_features/diarization_segments.json"
) -> Dict[str, Any]:
    """
    Perform speaker diarization on a mono podcast audio file using pyannote.audio.
    
    This function follows the exact specifications from task 2.1:
    - Load the pyannote speaker-diarization model
    - Process the mono audio file to identify speakers
    - Convert results to the specified JSON format
    
    Args:
        audio_file_path (str): Path to the input mono audio file
        output_file_path (str): Path to save the diarization results JSON
        
    Returns:
        dict: Diarization results containing segments and speaker information
    """
    
    print("🎙️  Starting speaker diarization on mono podcast...")
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        # Try to load from .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        except ImportError:
            pass
        
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found. Please set it in your environment.")
    
    print("✅ HuggingFace token found")
    
    try:
        # Import pyannote components
        print("📦 Loading pyannote.audio pipeline...")
        from pyannote.audio import Pipeline
        
        # Load the speaker diarization pipeline as specified in task 2.1
        # Use auth token parameter directly
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
                                          use_auth_token=hf_token)
        
        print(f"✅ Pipeline loaded successfully")
        print(f"🎵 Processing mono audio file: {audio_file_path}")
        
        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        # Run the pipeline on the mono audio file as specified
        print("🔍 Running speaker diarization...")
        diarization = pipeline(audio_file_path)
        
        print("📊 Converting diarization results...")
        
        # Convert diarization result into a list of segments as specified
        segments = []
        speaker_mapping = {}
        speaker_counter = 0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Map original speaker labels to simplified ones (S0, S1, S2, etc.)
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"S{speaker_counter}"
                speaker_counter += 1
            
            segment = {
                "speaker": speaker_mapping[speaker],
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            }
            segments.append(segment)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        # Create the exact JSON structure specified in task 2.1
        result = {
            "audio_file": audio_file_path,
            "segments": segments
        }
        
        # Ensure output directory exists
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON as specified
        print(f"💾 Saving diarization results to: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Perform the checks specified in task 2.1
        unique_speakers = list(set(seg["speaker"] for seg in segments))
        
        print("\n📊 Diarization Results (as specified in task 2.1):")
        print(f"   Total number of diarization segments: {len(segments)}")
        print(f"   Number of unique speakers: {len(unique_speakers)} ({', '.join(unique_speakers)})")
        
        print("\n🔍 Sample segments to verify speaker alternation:")
        for i, segment in enumerate(segments[:10]):
            print(f"     Segment {i+1}: {segment['speaker']} - {segment['start']:.2f}s to {segment['end']:.2f}s")
        
        if len(segments) > 10:
            print("     ...")
        
        # Additional analysis to verify alternation pattern
        speaker_changes = 0
        for i in range(1, len(segments)):
            if segments[i]["speaker"] != segments[i-1]["speaker"]:
                speaker_changes += 1
        
        print(f"\n✅ Speaker alternation analysis:")
        print(f"   Total speaker changes: {speaker_changes}")
        print(f"   Alternation rate: {(speaker_changes/len(segments)*100):.1f}% of segments")
        
        print("\n✅ Speaker diarization completed successfully!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during diarization: {str(e)}")
        raise


def main():
    """Main function to run diarization following task 2.1 specifications"""
    try:
        result = diarize_podcast()
        print(f"\n🎉 Task 2.1 completed! Diarization results saved to:")
        print("   outputs/audio_features/diarization_segments.json")
        
    except Exception as e:
        print(f"❌ Failed to complete task 2.1: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())