#!/usr/bin/env python3
"""
Script to count crash frames vs normal frames by analyzing CSV annotation files.
"""

import pandas as pd
import cv2
import os
import glob
from collections import defaultdict

def get_video_frame_count(video_path):
    """Get the total number of frames in a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return 0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return 0

def count_crash_frames(csv_path):
    """Count crash frames from CSV annotation file."""
    try:
        # Check if CSV has header
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Determine if header exists
        has_header = first_line.startswith('video_filename,frame_start_number')
        
        if has_header:
            df = pd.read_csv(csv_path)
        else:
            # Create DataFrame without header
            df = pd.read_csv(csv_path, header=None)
            df.columns = ['video_filename', 'frame_start_number', 'frame_end_number', 
                         'crash_flag', 'crash_position_x', 'crash_position_y', 'crash_severity']
        
        crash_frames = 0
        crash_ranges = []
        
        for _, row in df.iterrows():
            start_frame = int(row['frame_start_number'])
            end_frame = int(row['frame_end_number'])
            crash_frames += (end_frame - start_frame + 1)
            crash_ranges.append((start_frame, end_frame))
        
        return crash_frames, crash_ranges
    
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return 0, []

def main():
    """Main function to analyze all CSV files and count frames."""
    
    # Find all CSV annotation files
    csv_files = glob.glob("*_annotations.csv")
    
    if not csv_files:
        print("No annotation CSV files found!")
        return
    
    print(f"Found {len(csv_files)} annotation files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print()
    
    total_crash_frames = 0
    total_normal_frames = 0
    video_stats = {}
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        # Get crash frames from CSV
        crash_frames, crash_ranges = count_crash_frames(csv_file)
        
        # Extract video filename from CSV
        try:
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('video_filename'):
                # Has header, read second line for video name
                with open(csv_file, 'r') as f:
                    next(f)  # Skip header
                    second_line = f.readline().strip()
                    video_name = second_line.split(',')[0]
            else:
                # No header, first line has video name
                video_name = first_line.split(',')[0]
            
            # Check if video file exists
            if not os.path.exists(video_name):
                print(f"  Warning: Video file {video_name} not found!")
                continue
            
            # Get total frames from video
            total_frames = get_video_frame_count(video_name)
            normal_frames = total_frames - crash_frames
            
            if normal_frames < 0:
                print(f"  Warning: More crash frames ({crash_frames}) than total frames ({total_frames})!")
                normal_frames = 0
            
            # Store stats
            video_stats[video_name] = {
                'total_frames': total_frames,
                'crash_frames': crash_frames,
                'normal_frames': normal_frames,
                'crash_ranges': crash_ranges
            }
            
            total_crash_frames += crash_frames
            total_normal_frames += normal_frames
            
            print(f"  Video: {video_name}")
            print(f"  Total frames: {total_frames}")
            print(f"  Crash frames: {crash_frames}")
            print(f"  Normal frames: {normal_frames}")
            print(f"  Crash percentage: {(crash_frames/total_frames*100):.2f}%")
            print()
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            continue
    
    # Print summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total videos processed: {len(video_stats)}")
    print(f"Total crash frames: {total_crash_frames:,}")
    print(f"Total normal frames: {total_normal_frames:,}")
    print(f"Total frames: {total_crash_frames + total_normal_frames:,}")
    
    if total_crash_frames + total_normal_frames > 0:
        crash_percentage = (total_crash_frames / (total_crash_frames + total_normal_frames)) * 100
        print(f"Overall crash percentage: {crash_percentage:.2f}%")
    
    print("\nPer-video breakdown:")
    for video_name, stats in video_stats.items():
        print(f"  {video_name}:")
        print(f"    Crash frames: {stats['crash_frames']:,}")
        print(f"    Normal frames: {stats['normal_frames']:,}")
        print(f"    Total frames: {stats['total_frames']:,}")

if __name__ == "__main__":
    main() 