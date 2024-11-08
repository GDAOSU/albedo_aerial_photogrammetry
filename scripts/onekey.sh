#!/bin/bash
export workdir=$1
echo $workdir
if [ -z "$workdir" ]; then
    echo "Usage: $0 <workdir>"
    exit 1
fi

python step1_pull_images.py --nlevel 2 $workdir
./step2.1_ambient_occ_baker.sh $workdir
./step2.2_render_buffers.sh $workdir 2
python step3_crf_sunvis_refinement.py $workdir
python step4_sample_litshadow_band.py $workdir
python step5_process_litshadow_band.py $workdir
python step6_compute_sunsky_ratio.py --skymodel Simple --logarithm_phi $workdir
python step7_decompose_albedo.py --median_phi $workdir $workdir/export_simple_albedo

python step8_undistortion.py $workdir/imagedataset.json $workdir/image $workdir/undist_simple_albedo/rgb
python step8_undistortion.py $workdir/imagedataset.json $workdir/normal $workdir/undist_simple_albedo/normal
python step8_undistortion.py $workdir/imagedataset.json $workdir/depth $workdir/undist_simple_albedo/depth
python step8_undistortion.py $workdir/imagedataset.json $workdir/export_simple_albedo/albedo $workdir/undist_simple_albedo/albedo
python step8_undistortion.py $workdir/imagedataset.json $workdir/export_simple_albedo/mask $workdir/undist_simple_albedo/mask
python step8_undistortion.py $workdir/imagedataset.json $workdir/export_simple_albedo/sunvis $workdir/undist_simple_albedo/sunvis
python step8_undistortion.py $workdir/imagedataset.json $workdir/export_simple_albedo/sun_shading $workdir/undist_simple_albedo/sun_shading
python step8_undistortion.py $workdir/imagedataset.json $workdir/export_simple_albedo/sky_shading $workdir/undist_simple_albedo/sky_shading

python step9_hdr2ldr.py $workdir/undist_simple_albedo/rgb $workdir/undist_simple_albedo/rgb_ldr
python step9_hdr2ldr.py $workdir/undist_simple_albedo/albedo $workdir/undist_simple_albedo/albedo_ldr
python step9_hdr2ldr.py $workdir/undist_simple_albedo/mask $workdir/undist_simple_albedo/mask_ldr --min_perc 0 --max_perc 100 

python step10_export_cam.py $workdir/imagedataset.json $workdir/undist_simple_albedo/pose --undist --format ".json"
python step10_export_cam.py $workdir/imagedataset.json $workdir/undist_simple_albedo/odmcam --undist --format ".png.cam"
