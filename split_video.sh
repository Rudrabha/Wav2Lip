#!/bin/bash

# Function to split video using ffmpeg
split_video() {
    local input_file=$1
    local split_duration=$2
    local output_prefix=$3

    # Get duration of video in seconds
    local duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file")
    local duration=${duration%.*} # Remove decimal part

    # Calculate number of parts we need to split into based on duration and split_duration
    if (( duration > split_duration )); then
        
        # Split the video into segments of split_duration each
        ffmpeg -i "$input_file" -c copy -map 0 -segment_time "$split_duration" -f segment -reset_timestamps 1 "${output_prefix}_%04d.${input_file##*.}"
        echo "Video $input_file has been split into multiple parts."
    else
        echo "Video $input_file does not require splitting."
    fi
}

# Check if at least two arguments are provided (split_duration and base_dir)
if [[ $# -lt 2 ]]; then
    echo "Usage: $0  "
    exit 1
fi

split_duration="$1"
base_dir="$2"

# Ensure provided base_dir exists and is a directory.
if [[ ! -d "$base_dir" ]]; then
  echo "The specified base directory does not exist or is not a directory."
  exit 2
fi

# Define an array of video file extensions.
video_extensions=("mp4" "mkv" "avi" "mov") # Add any other video formats you need.

# Build find command expression.
find_expression=()
for ext in "${video_extensions[@]}"; do
  find_expression+=(-iname "*.$ext" -o)
done
# Remove trailing -o (logical OR).
last_index=$(( ${#find_expression[@]} - 1 ))
unset 'find_expression[last_index]'

# Find all video files in base_dir and its subdirectories.
while IFS= read -r file; do
    
    # Generate output prefix by removing file extension from input filename.
    output_prefix="${file%.*}"

    echo 'start processing' $file $output_prefix
    # Call function to split video.
    split_video "$file" "$split_duration" "$output_prefix"

done < <(find "$base_dir" -type f \( "${find_expression[@]}" \))

echo "All videos processed."