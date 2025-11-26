#!/usr/bin/env python3
"""
Update Speaker Labels Script

This script updates the speaker labels in the diarization JSON file:
- S2 -> Joe Rogan
- S0, S1, S3 -> Donald Trump
"""

import json
import sys
from pathlib import Path

def update_speaker_labels(input_file, output_file=None):
    """
    Update speaker labels in diarization JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (if None, overwrites input)
    """
    
    # If no output file specified, overwrite the input file
    if output_file is None:
        output_file = input_file
    
    print(f"📖 Reading diarization data from: {input_file}")
    
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count original speaker occurrences
    speaker_counts = {}
    for segment in data['segments']:
        speaker = segment['speaker']
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    print(f"📊 Original speaker distribution:")
    for speaker, count in sorted(speaker_counts.items()):
        print(f"   {speaker}: {count} segments")
    
    # Update speaker labels according to the mapping
    updated_count = 0
    speaker_mapping = {
        'S0': 'Donald Trump',
        'S1': 'Donald Trump', 
        'S2': 'Joe Rogan',
        'S3': 'Donald Trump'
    }
    
    print(f"🔄 Applying speaker label mapping:")
    for old_label, new_label in speaker_mapping.items():
        print(f"   {old_label} -> {new_label}")
    
    # Apply the mapping
    for segment in data['segments']:
        old_speaker = segment['speaker']
        if old_speaker in speaker_mapping:
            segment['speaker'] = speaker_mapping[old_speaker]
            updated_count += 1
    
    # Count new speaker occurrences
    new_speaker_counts = {}
    for segment in data['segments']:
        speaker = segment['speaker']
        new_speaker_counts[speaker] = new_speaker_counts.get(speaker, 0) + 1
    
    print(f"✅ Updated {updated_count} segments")
    print(f"📊 New speaker distribution:")
    for speaker, count in sorted(new_speaker_counts.items()):
        print(f"   {speaker}: {count} segments")
    
    # Save the updated data
    print(f"💾 Saving updated data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 Speaker labels updated successfully!")
    
    return data

def main():
    """Main function to run the speaker label update."""
    
    # Default file path
    input_file = "outputs/audio_features/diarization_segments.json"
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"❌ Error: File not found: {input_file}")
        print(f"   Please run this script from the podcast_analysis directory.")
        return 1
    
    try:
        # Update the speaker labels
        update_speaker_labels(input_file)
        return 0
        
    except Exception as e:
        print(f"❌ Error updating speaker labels: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())