python external/vggt/demo_colmap.py --scene_dir=colmap_project_dir --use_ba --max_query_pts=512 --query_frame_num=4 --vis_thresh=0.5 --fine_tracking

python external/vggt/demo_colmap.py --scene_dir=colmap_project_dir --use_ba --max_query_pts=512 --query_frame_num=8 --vis_thresh=0.5 --fine_tracking

python external/vggt/demo_colmap.py --scene_dir=colmap_project_dir --use_ba --max_query_pts=512 --query_frame_num=8 --vis_thresh=0.5 --max_reproj_error=4.0 --fine_tracking

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python external/vggt/demo_colmap.py --scene_dir=colmap_project_dir --use_ba --max_query_pts=320 --query_frame_num=4 --vis_thresh=0.8 --max_reproj_error=2.5

images = bbox_with_alpha
masks = bbox_masks