# Mask Builder

Most open source tools for image segmentation annotations are closed source or require extensive installations.  
This is an extremely minimal tool to allow for creating and saving masks.

## Usage

```bash
python create_mask.py --img_name ./test_image.jpg --save_dir ./ --mask_color 0,0,0 --save_fname test_mask.csv
```

## Improvements
* Replace Convexhull with concave hull algorithm