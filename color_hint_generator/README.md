# Color Hint Generator

We developed a color hint generator that selects a single pixel's color from each segment of a segmented colored sketch. Using these selected colors, it generates a color hint image to guide further processing or visualization.

## Requirements

python3

python library:<br>
numpy<br>
cv2

## Usage

Put the generated draft into the post1 or post2 folder, then put its segmented image into the according segment1 or segment2 folder, then run the generator code, the corresponding color hint image will be generate in the color_hint1 or color_hint2 folder.

## Example

| Draft           | Segment         | Color Hint      |
|------------------|-----------------|-----------------|
| ![Draft](https://github.com/Tyi77/comic-colorization/blob/Jerry/color_hint_generator/post1/1_100.jpg) | ![Segment](https://github.com/Tyi77/comic-colorization/blob/Jerry/color_hint_generator/segment1/1.jpg) | ![Color Hint](https://github.com/Tyi77/comic-colorization/blob/Jerry/color_hint_generator/color_hints1/color_hint1.png) |

