# Gaussian Splatting with Sparse View Directions

This project is still ongoing and thus the repository is still under construction.

## Misc Development Notes
### Preprocessing in Maya

- Cameras 03-07 in Penn's Captury rig are flipped upside down.
- This MEL code block in Maya rotates those cameras 180 degrees on their local z-axis.

```mel
string $cams[] = {"cam03", "cam04", "cam05", "cam06", "cam07"};
for ($cam in $cams) {
    rotate -r -os 0 0 180 $cam;
};
```

### Using uv python manager

- Run python scripts with env variables.

```bash
uv run --env-file .env .\src\vggt\demo_colmap.py --scene_dir .\resources\colmap_project_dir\
```

- Can also look into using a custom shebang like so to simplify things:

```bash
#!/usr/bin/env -S uv run --env-file .env
```

Then `.env` file will look something like:

```bash
VGGT_CACHE_DIR="/home/aliu/.cache/vggt"
```

To sync up PyPI packages:

```bash
uv sync --group vggt
```

Haven't figured out how to also get an additional dependency group working with different versions of a package, but when I do it will change to something like:

```bash
uv sync --group gsplat --no-group=vggt
```

### demo_colmap trials

```bash
python src/vggt/demo_colmap.py --scene_dir=resources/colmap_project_dir --use_ba --max_query_pts=1024 --query_frame_num=8
```
