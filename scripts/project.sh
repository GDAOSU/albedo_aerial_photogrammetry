
#coe_corridor osu_office <SRS>ENU:40.00231,-83.01576</SRS>
#20250314_am_sunny#
#20250321_cloudy#
#20250327_pm_sunny#
#goodale_park public_park
#20250312_am_sunny#
#20250416_pm_sunny#
#20250425_cloudy#
#jesse_owens_residential osu_residence
#20250416_pm_sunny#
#20250424_am_sunny#
#20250425_cloudy#
#schottenstein_center schottenstein
#20250314_am_sunny#
#20250321_pm_sunny#
#20250326_cloudy

export workspace="schottenstein"
export uav_workspace="schottenstein_center"
export project="20250326_cloudy"

export project_dir="/research/GDA/data/UAV_images/Multitemporal_Dataset/2025_result/Model/${workspace}"
export image_dir="/research/GDA/data/UAV_images/Multitemporal_Dataset/2025_Multitemporal_Dataset/${uav_workspace}/${project}/"

export workdir="/research/GDA/data/UAV_images/Multitemporal_Dataset/2025_result/Albedo/${workspace}/${project}"
export block_xml="${project_dir}/${project}_export.xml"
export obj_dir="$project_dir/Productions/${project}/"
export nLevel=3

python step0_blockxml_to_imagedataset.py $block_xml $workdir/imagedataset.json --default_redirect $image_dir
python step0_tiledobj_to_modeldataset.py $obj_dir $workdir/modeldataset.json
python step1_pull_images.py  $workdir --nlevel $nLevel

./step2.1_ambient_occ_baker.sh
./step2.2_render_buffers.sh
python step3_crf_sunvis_refinement.py $workdir
python step4_sample_litshadow_band.py $workdir
python step5_process_litshadow_band.py $workdir
python step6_compute_sunsky_ratio.py --skymodel AO --logarithm_phi $workdir 
python step7_decompose_albedo.py --median_phi $workdir $workdir/export
python step8_undistortion.py $workdir/imagedataset.json  $workdir/export --gammar_correction

