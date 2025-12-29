.PHONY: all

process_frames:
	python scripts/avi_to_frames.py resources/raw_data/caroline_shot --start_frame=0 --end_frame=24

process_backgrounds:
	python scripts/avi_to_frames.py resources/raw_data/empty_shot --start_frame=0 --end_frame=24

rotate_frames:
	python scripts/preprocess.py --source_dir=resources/raw_data/caroline_shot --background_dir=resources/raw_data/empty_shot --start_frame=0 --end_frame=24

segment_frames:
	python scripts/mask.py --source_dir=resources/raw_data/caroline_shot --background_dir=resources/raw_data/empty_shot --start_frame=0 --end_frame=24