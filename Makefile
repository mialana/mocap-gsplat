.PHONY: all

clone:
	git clone --recurse-submodules repo_url

viser:
	python external/vggt/demo_viser.py --image_folder colmap_project_dir/images

gradio:
	python external/vggt/demo_gradio.py

colmap:
	python external/vggt/demo_colmap.py --scene_dir=colmap_project_dir --use_ba --max_query_pts=1024 --query_frame_num=4

extra-installs:
	pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe

train:
	python external/gsplat/examples/simple_trainer.py default --disable_video --data_factor 1 --data_dir colmap_project_dir --result_dir RESULTS --save_ply