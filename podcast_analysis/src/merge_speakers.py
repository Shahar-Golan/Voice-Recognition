"""
Speaker-ASR Merge Module

This module implements task 4.1 from new_way.md:
- Merge diarization segments (who spoke when) with ASR transcript (what was said when)
- Use robust overlap-based algorithm to assign speakers to words
- Generate final transcript with speaker labels and word-level timestamps
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def assign_speaker_to_word(word: Dict[str, Any], diar_segments: List[Dict[str, Any]]) -> Optional[str]:
    """
    Assign a speaker to a word based on temporal overlap with diarization segments.
    
    This robust algorithm finds the diarization segment with maximum overlap
    with the word's timestamp interval.
    
    Args:
        word (dict): Word with "start" and "end" timestamps
        diar_segments (list): List of diarization segments with speaker, start, end
        
    Returns:
        str: Speaker label (e.g., "Joe Rogan", "Donald Trump") or None if no overlap
    """
    w_start, w_end = word["start"], word["end"]
    best_speaker = None
    best_overlap = 0.0

    for seg in diar_segments:
        d_start, d_end = seg["start"], seg["end"]

        # Calculate intersection of [w_start, w_end] and [d_start, d_end]
        overlap = max(0.0, min(w_end, d_end) - max(w_start, d_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def merge_diarization_and_asr(
    diarization_file: str = "outputs/audio_features/diarization_segments.json",
    transcript_file: str = "outputs/audio_features/transcript_words.json",
    output_file: str = "outputs/audio_features/transcript_with_speakers.json"
) -> Dict[str, Any]:
    """
    Merge diarization and ASR results into speaker-labeled transcript.
    
    Implements task 4.1 specifications:
    - Combines speaker diarization with word-level ASR timestamps
    - Uses robust overlap algorithm for speaker assignment
    - Generates final transcript in specified JSON format
    
    Args:
        diarization_file (str): Path to diarization segments JSON
        transcript_file (str): Path to ASR transcript JSON
        output_file (str): Path to save merged transcript
        
    Returns:
        dict: Merged transcript data
    """
    
    print("🔀 Starting diarization + ASR merge...")
    print(f"📂 Diarization: {diarization_file}")
    print(f"📂 Transcript: {transcript_file}")
    print(f"📄 Output: {output_file}")
    
    # Load diarization segments
    print("📖 Loading diarization segments...")
    if not os.path.exists(diarization_file):
        raise FileNotFoundError(f"Diarization file not found: {diarization_file}")
    
    with open(diarization_file, 'r', encoding='utf-8') as f:
        diar_data = json.load(f)
    
    diar_segments = diar_data["segments"]
    print(f"✅ Loaded {len(diar_segments)} diarization segments")
    
    # Load ASR transcript
    print("📖 Loading ASR transcript...")
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        asr_data = json.load(f)
    
    asr_segments = asr_data["segments"]
    print(f"✅ Loaded {len(asr_segments)} ASR segments")
    
    # Collect all words with timestamps
    print("🔄 Collecting all words with timestamps...")
    all_words = []
    
    for asr_segment in asr_segments:
        if "words" in asr_segment:
            for word in asr_segment["words"]:
                if word.get("word", "").strip():  # Skip empty words
                    all_words.append({
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "word": word["word"].strip()
                    })
    
    print(f"✅ Collected {len(all_words)} words with timestamps")
    
    # Assign speakers to all words using robust overlap algorithm
    print("🎯 Assigning speakers to words using overlap algorithm...")
    words_with_speakers = []
    
    for word in all_words:
        speaker = assign_speaker_to_word(word, diar_segments)
        if speaker:  # Only include words that have speaker assignments
            words_with_speakers.append({
                "start": word["start"],
                "end": word["end"], 
                "word": word["word"],
                "speaker": speaker
            })
    
    print(f"✅ Assigned speakers to {len(words_with_speakers)} words")
    
    # Group words by speaker segments according to diarization timing
    print("📝 Building speaker-labeled segments...")
    final_segments = []
    
    for diar_seg in diar_segments:
        # Find all words that overlap with this diarization segment
        segment_words = []
        
        for word in words_with_speakers:
            if word["speaker"] == diar_seg["speaker"]:
                # Check if word overlaps with diarization segment
                overlap = max(0.0, min(word["end"], diar_seg["end"]) - max(word["start"], diar_seg["start"]))
                if overlap > 0:
                    segment_words.append(word)
        
        # Sort words by start time
        segment_words.sort(key=lambda w: w["start"])
        
        if segment_words:  # Only create segments that have words
            # Build text from words
            text = " ".join([w["word"] for w in segment_words])
            
            # Create final segment structure as specified in task 4.1
            final_segment = {
                "speaker": diar_seg["speaker"],
                "start": float(diar_seg["start"]),
                "end": float(diar_seg["end"]),
                "text": text.strip(),
                "words": [
                    {
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["word"]
                    }
                    for w in segment_words
                ]
            }
            
            final_segments.append(final_segment)
    
    # Get unique speakers for the output
    unique_speakers = list(set(seg["speaker"] for seg in final_segments))
    unique_speakers.sort()
    
    # Create final merged transcript in exact format specified
    merged_transcript = {
        "audio_file": asr_data.get("audio_file", "data/processed/podcast_16k_mono.wav"),
        "sample_rate": asr_data.get("sample_rate", 16000),
        "speakers": unique_speakers,
        "segments": final_segments
    }
    
    # Ensure output directory exists
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged transcript
    print(f"💾 Saving merged transcript to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_transcript, f, indent=2, ensure_ascii=False)
    
    # Print required checks from task 4.1
    print("\n📊 Merge Results (as specified in task 4.1):")
    print(f"   Number of final segments: {len(final_segments)}")
    
    # Count segments per speaker
    speaker_counts = {}
    speaker_durations = {}
    
    for segment in final_segments:
        speaker = segment["speaker"]
        duration = segment["end"] - segment["start"]
        
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
    
    print(f"   Unique speakers and segment counts:")
    for speaker in sorted(speaker_counts.keys()):
        print(f"     {speaker}: {speaker_counts[speaker]} segments")
    
    # Compute speaking time distribution
    total_duration = sum(speaker_durations.values())
    print(f"\n🎯 Conversational distribution:")
    for speaker in sorted(speaker_durations.keys()):
        percentage = (speaker_durations[speaker] / total_duration) * 100 if total_duration > 0 else 0
        print(f"   {speaker}: {percentage:.1f}% of time ({speaker_durations[speaker]:.1f}s)")
    
    # Show sample segments for manual inspection
    print(f"\n🔍 Sample segments for manual inspection:")
    for i, segment in enumerate(final_segments[:5]):
        print(f"     Segment {i+1}: [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}")
        text_preview = segment['text'][:80] + "..." if len(segment['text']) > 80 else segment['text']
        print(f"       Text: \"{text_preview}\"")
        print(f"       Words: {len(segment['words'])} words")
    
    if len(final_segments) > 5:
        print("     ...")
    
    print(f"\n✅ Task 4.1 completed! Merged transcript saved to: {output_file}")
    
    return merged_transcript


def main():
    """Main function to run speaker-ASR merging following task 4.1"""
    try:
        result = merge_diarization_and_asr()
        print(f"\n🎉 Speaker-ASR merge completed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to complete task 4.1: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())