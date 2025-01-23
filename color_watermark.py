import cv2
import numpy as np
from skimage import io
import os

# import matplotlib.pyplot as plt


def get_color(img):
    pixels = np.float32(img.reshape(-1, 3))

    average = img.mean(axis=0).mean(axis=0)

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    # print(counts)
    dominant = palette[np.argmax(counts)]
    # print(palette)
    # print(f"dominant color: {dominant}")
    return average, dominant, palette, counts


# Sum of the min & max of (a, b, c)
def hilo(a, b, c):
    if c < b:
        b, c = c, b
    if b < a:
        a, b = b, a
    if c < b:
        b, c = c, b
    return a + c


def complement(color):
    r, g, b = color[0], color[1], color[2]
    if r + g + b < 50:
        return [255 - r, 255 - g, 255 - b]
    k = hilo(r, g, b)
    return [k - u for u in (r, g, b)]


def create_colored_obj(input_obj_path, color, out_dir, suffix='clr', remove=True):
    # Color must be a tuple (R, G, B) in the range 0-255
    r, g, b = color
    mtl_content = f"""
    newmtl colored_material
    Kd {r/255} {g/255} {b/255}
    """

    # filename = input_obj_path.split("/")[-1][:-4]
    filename = os.path.splitext(os.path.basename(input_obj_path))[0]
    output_obj_path = os.path.join(out_dir, f"{filename}_{suffix}.obj")
    output_mtl_path = os.path.join(out_dir, f"{filename}_{suffix}.mtl")

    with open(output_mtl_path, "w") as mtl_file:
        mtl_file.write(mtl_content)
        # print(f"Exported mtl {output_mtl_path}")

    with open(input_obj_path, 'r') as temp_file:
        lines = temp_file.readlines()

    with open(output_obj_path, "w") as new_obj_file:
        new_obj_file.write(f"mtllib {output_mtl_path.split('/')[-1]}\n")
        for line in lines:
            if line.startswith("usemtl") or line.startswith("mtllib"):
                continue
            if line.startswith("f "):  # Before the first face, reference the new material
                new_obj_file.write("usemtl colored_material\n")
            new_obj_file.write(line)
    # print(f"Exported colored obj {output_obj_path}")

    # print(f"Removing {input_obj_path}")
    if remove:
        os.remove(input_obj_path)

    return output_obj_path, output_mtl_path