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