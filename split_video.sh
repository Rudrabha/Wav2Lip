#!/bin/bash

# Directory containing the original MP4 files
SOURCE_DIR="/path/to/source"

# Function to split video into two halves if longer than one minute
split_video_if_longer_than_one_minute() {
    local input_file="$1"
    local output_dir="$2"
    local file_index="$3"
    
    # Check if the video is longer than 60 seconds (1 minute)
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file")
    
    if (( $(echo "$duration > 60" | bc -l) )); then
        # Calculate half the duration
        half_duration=$(echo "$duration / 2" | bc -l)

        # Create subfolder for split videos if it doesn't exist
        mkdir -p "$output_dir"

        # Split the video into two halves using FFmpeg
        ffmpeg -i "$input_file" -t "$half_duration" -c copy "${output_dir}/0000${file_index}.mp4"
        let file_index=file_index+1
        ffmpeg -i "$input_file" -ss "$half_duration" -c copy "${output_dir}/0000${file_index}.mp4"
    fi
}

# Initialize file index counter
file_index=1

# Export function so it can be used by find command with exec option
export -f split_video_if_longer_than_one_minute

# Iterate over all MP4 files in the directory and its subdirectories,
# splitting them if they are longer than one minute.
find "$SOURCE_DIR" -type f -name "*.mp4" | while read filename; do 
    split_video_if_longer_than_one_minute "$filename" "$(dirname "$filename")/split" $file_index
    
    # Increment index only if a file was actually split.
    if [ $? == 0 ]; then 
        let file_index=file_index+2 
    fi 
done
