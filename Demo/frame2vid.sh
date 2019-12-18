#!/bin/bash

#ffmpeg -framerate 10 -i %03d.jpeg output_no_audio.mp4
#ffmpeg -i test1.mp3 -ac 2 output_stereo.mp3

#ffmpeg -i output_no_audio.mp4 -i output_stereo.mp3 -c:v copy -map 0:v:0 -map 1:a:0 -strict -2 output_with_audio.mp4

# audio_dir="/home/xuanchi/August/train_for_boy/pose2vid/result/fake/audio"
# frame_dir="/home/xuanchi/August/train_for_boy/pose2vid/result/fake"
# audio_dir="/home/xuanchi/user_study/local_GCN_perceptual_D_Feature_real/audio"
# frame_dir="/home/xuanchi/user_study/local_GCN_perceptual_D_Feature_real/"
audio_dir="/home/xuanchi/self_attention_model/test_different_music/anymore/audio"
frame_dir="/home/xuanchi/self_attention_model/test_different_music/anymore/"

copy_dir="$( printf "%s/videos" $frame_dir )"

mkdir "$copy_dir"

#how many video you want to generate, the max of this variable is the numbers of the output folder
max=35

for (( i=0; i <= $max; ++i ))
do
    fname="$( printf "%s/%d" $frame_dir $i )"
    cp_dir="$( printf "%s/%d.mp4" $copy_dir $i )"
    cd $fname
    ffmpeg -y -framerate 10 -i %05d.jpeg output_no_audio.mp4

    audio_name="$( printf "%s/%d.wav" $audio_dir $i )"

    echo "$fname"
    echo "$audio_name"

    ffmpeg -y -i "$audio_name" -ac 2 output_stereo.mp3
    ffmpeg -y -i output_no_audio.mp4 -i output_stereo.mp3 -c:v copy -map 0:v:0 -map 1:a:0 -strict -2 output_with_audio.mp4
    
    
    cp output_with_audio.mp4 "$cp_dir"

done