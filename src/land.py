import sys
import json

from PIL import Image, ImageChops
import numpy as np


def main():
    height_map_image = Image.open(sys.argv[1])
    print(f"Height map image size: {height_map_image.size}")
    # height_map_image_rgb = height_map_image.convert('RGB')
    # height_map_matrix = np.array(height_map_image)
    # registry = json.loads(open(sys.argv[2], 'rt').read())
    registry = dict(
    )
    land_types = {}
    # land_image = Image.new("RGB", (height_map_image.width, height_map_image.height), "black")
    min_height = None
    max_height = None
    for y in range(height_map_image.height):
        for x in range(height_map_image.width):
            height = float(height_map_image.getpixel((x, y)))
            if min_height is None or height < min_height:
                min_height = height
            if max_height is None or height > max_height:
                max_height = height
    print(f"Min height: {min_height}, Max height: {max_height}")
    


if __name__ == '__main__':
    main()
