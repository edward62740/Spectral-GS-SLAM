OUTPUT_PATH="${OUTPUT_PATH:-experiments/results}"
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to the directory containing rgbd_dataset_freiburg1_desk/, ...}"

str_pad() {

  local pad_length="$1" pad_string="$2" pad_type="$3"
  local pad length llength offset rlength

  pad="$(eval "printf '%0.${#pad_string}s' '${pad_string}'{1..$pad_length}")"
  pad="${pad:0:$pad_length}"

  if [[ "$pad_type" == "left" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "${pad:0:$length}$line"
    done

  elif [[ "$pad_type" == "both" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      llength="$(( length / 2 ))"
      offset="$(( llength + ${#line} ))"
      rlength="$(( llength + (length % 2) ))"
      echo -n "${pad:0:$llength}$line${pad:$offset:$rlength}"
    done

  else

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "$line${pad:${#line}:$length}"
    done

  fi
}

run_()
{
    local dataset=$1
    local config=$2
    local result_txt=$3
    local keyframe_th=$4
    local knn_maxd=$5
    local overlapped_th=$6
    local max_correspondence_distance=$7
    local trackable_opacity_th=$8
    local overlapped_th2=$9
    local downsample_rate=${10}
    
    echo "run $dataset"
    python3 -W ignore gs_icp_slam.py  --dataset_path $DATASET_PATH/$dataset\
                                    --config $config\
                                    --output_path $OUTPUT_PATH\
                                    --keyframe_th $keyframe_th\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --save_results\
                                    --verbose\
                                    --use_fusion
                                    

    wait
}

run_tum()
{
    local result_txt=$1
    local keyframe_th=$2
    local knn_maxd=$3
    local overlapped_th=$4
    local max_correspondence_distance=$5
    local trackable_opacity_th=$6
    local overlapped_th2=$7
    local downsample_rate=$8

    # same intrinsic params for all
    #run_ "rgbd_dataset_freiburg1_desk" "configs/TUM/rgbd_dataset_freiburg1_desk.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg1_room" "configs/TUM/rgbd_dataset_freiburg1_desk.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate 
    #run_ "rgbd_dataset_freiburg1_xyz" "configs/TUM/rgbd_dataset_freiburg1_desk.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg2_desk" "configs/TUM/rgbd_dataset_freiburg1_desk.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg2_desk2" "configs/TUM/rgbd_dataset_freiburg1_desk.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg2_large_no_loop" "configs/TUM/rgbd_dataset_freiburg2_large_no_loop.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg2_xyz" "configs/TUM/rgbd_dataset_freiburg2_xyz.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_long_office_household" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_walking_static" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_sitting_rpy" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_walking_rpy" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_walking_halfsphere" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_walking_xyz" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_nostructure_notexture_far" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_structure_notexture_near" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_structure_notexture_far" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_structure_notexture_near_validation" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_structure_notexture_far_validation" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate

    #run_ "rgbd_dataset_freiburg3_nostructure_texture_near_withloop" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "rgbd_dataset_freiburg3_nostructure_texture_far" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_nostructure_texture_far_validation" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_nostructure_texture_near_withloop_validation" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_large_cabinet" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
  
    #run_ "rgbd_dataset_freiburg3_sitting_static" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    #run_ "rgbd_dataset_freiburg3_sitting_xyz" "configs/TUM/rgbd_dataset_freiburg3_long_office_household.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
}

run_replica()
{
    local result_txt=$1
    local keyframe_th=$2
    local knn_maxd=$3
    local overlapped_th=$4
    local max_correspondence_distance=$5
    local trackable_opacity_th=$6
    local overlapped_th2=$7
    local downsample_rate=$8

    run_ "room0" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "room1" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "room2" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "office0" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "office1" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "office2" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "office3" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "office4" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
}

run_scannet()
{
    local result_txt=$1
    local keyframe_th=$2
    local knn_maxd=$3
    local overlapped_th=$4
    local max_correspondence_distance=$5
    local trackable_opacity_th=$6
    local overlapped_th2=$7
    local downsample_rate=$8

    run_ "8b5caf3398" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
    run_ "b20a261fdf" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate
}

# txt_file="re_init_ablation/default_3DGS.txt"
txt_file="plane_regularization.txt"
str_pad 20 " " left <<< "FPS" > $txt_file
str_pad 15 " " left <<< "RMSE" >> $txt_file
str_pad 15 " " left <<< "train iter" >> $txt_file
str_pad 15 " " left <<< "kframes" >> $txt_file
str_pad 15 " " left <<< "gaussians_num" >> $txt_file
# str_pad 32 " " left <<< "Depth L1" >> $txt_file
str_pad 30 " " left <<< "PSNR" >> $txt_file
str_pad 15 " " left <<< "SSIM" >> $txt_file
str_pad 15 " " left <<< "LPIPS" >> $txt_file
echo "" >> $txt_file

overlapped_th=1e-3
max_correspondence_distance=0.05
knn_maxd=99999.0

trackable_opacity_th=0.09
overlapped_th2=1e-3
downsample_rate=5
keyframe_th=0.81

run_tum $txt_file $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
$trackable_opacity_th $overlapped_th2 $downsample_rate
