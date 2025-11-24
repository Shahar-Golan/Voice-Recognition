"""
Configuration file for podcast audio analysis pipeline.

This module defines all the constants and configuration parameters
used throughout the audio feature extraction and analysis process.
"""

# Audio file paths
AUDIO_RAW_PATH = "data/raw/podcast.mp3"
AUDIO_WAV_PATH = "data/processed/podcast_16k_mono.wav"

# Audio processing parameters
SAMPLE_RATE = 16000
WINDOW_SIZE_SEC = 5.0

# Output file paths
AUDIO_FEATURES_CSV = "outputs/audio_features/audio_features_5s_windows.csv"
AUDIO_FEATURES_FIRED_CSV = "outputs/audio_features/audio_features_5s_windows_with_fired.csv"

# Analysis parameters
FIRED_Z_THRESHOLD = 2.0  # example threshold