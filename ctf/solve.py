#!/usr/bin/env python3
"""
Labels every image in pictures/ as 'hotdog' or 'not hotdog' with the
frozen retrained-Inception graph (graph_hotdog.pb), writes a black/white
100x100 tile per image into processed_2/, and reassembles the tiles into
the QR code they were rendered from.
"""
import os
import sys

import numpy as np
import tensorflow as tf
import imageio.v3 as iio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_LAYER = 'DecodeJpeg/contents:0'
OUTPUT_LAYER = 'final_result:0'

PICTURES_DIR = 'pictures'
PROCESSED_DIR = 'processed'
GRAPH_PATH = 'graph_hotdog.pb'
LABELS_PATH = 'labels_hotdog.txt'

TILE_SIZE = 100          # size of each written processed_2/ tile, as in the original script
QR_OUTPUT = 'qrcode.png'
MODULE_PX = 10            # pixel size of one QR module in the assembled image
QUIET_MODULES = 4         # white quiet-zone border a QR scanner expects
REDUNDANCY = 3            # each real QR module was photographed as a REDUNDANCYxREDUNDANCY
                           # block of images (87x87 images -> 29x29 QR modules), so a
                           # majority vote per block corrects individual misclassifications


def load_graph():
    with tf.io.gfile.GFile(GRAPH_PATH, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def classify_all():
    labels = [line.rstrip() for line in tf.io.gfile.GFile(LABELS_PATH)]
    load_graph()

    filenames = sorted(f for f in os.listdir(PICTURES_DIR) if f.lower().endswith('.jpg'))
    if not filenames:
        sys.exit(f'no images found in {PICTURES_DIR}/')

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    is_hotdog = []
    with tf.compat.v1.Session() as sess:
        soft_tensor = sess.graph.get_tensor_by_name(OUTPUT_LAYER)
        for i, filename in enumerate(filenames):
            image = tf.io.gfile.GFile(os.path.join(PICTURES_DIR, filename), 'rb').read()
            predict, = sess.run(soft_tensor, {INPUT_LAYER: image})
            result = labels[int(np.argmax(predict))]
            hotdog = result == 'hotdog'
            is_hotdog.append(hotdog)

            tile = np.zeros([TILE_SIZE, TILE_SIZE, 3], dtype=np.uint8)
            if hotdog:
                tile.fill(255)
            iio.imwrite(os.path.join(PROCESSED_DIR, filename), tile)

            if (i + 1) % 200 == 0 or i + 1 == len(filenames):
                print(f'[{i + 1}/{len(filenames)}] {filename}: {result}')

    return filenames, is_hotdog


def finder_pattern():
    """The standard 7x7 QR finder pattern (True = dark module)."""
    p = np.zeros((7, 7), dtype=bool)
    p[0, :] = p[6, :] = p[:, 0] = p[:, 6] = True  # outer black ring
    p[2:5, 2:5] = True                             # inner black 3x3
    return p


FINDER_PATTERN = finder_pattern()


def stamp_finder_patterns(dark, side):
    """Overlay the three fixed finder patterns (+ white separator) on a
    dark/light module grid. These are always identical on any QR code, so
    they don't need to be read from the photos at all - stamping the known
    pattern is more reliable than trusting the classifier in exactly the
    spot a scanner looks at first."""
    d = dark.copy()
    for r0, c0 in ((0, 0), (0, side - 7), (side - 7, 0)):
        d[r0:r0 + 7, c0:c0 + 7] = FINDER_PATTERN
    # one-module white separator between each finder pattern and the data area
    d[7, 0:8] = False
    d[0:8, 7] = False
    d[7, side - 8:side] = False
    d[0:8, side - 8] = False
    d[side - 8, 0:8] = False
    d[side - 8:side, 7] = False
    return d


def build_canvas(is_hotdog, grid_side, transposed, inverted):
    grid = np.array(is_hotdog, dtype=bool).reshape(grid_side, grid_side)

    # each real QR module was rendered as a REDUNDANCYxREDUNDANCY block of
    # photos; collapse each block to a single module by majority vote so
    # individual misclassified photos don't corrupt the code
    side = grid_side // REDUNDANCY
    blocks = grid.reshape(side, REDUNDANCY, side, REDUNDANCY)
    hotdog_majority = blocks.sum(axis=(1, 3)) >= (REDUNDANCY * REDUNDANCY + 1) // 2

    if transposed:
        hotdog_majority = hotdog_majority.T
    # original per-image logic: hotdog -> white(255), not hotdog -> black(0)
    dark = hotdog_majority if inverted else ~hotdog_majority
    dark = stamp_finder_patterns(dark, side)

    img_side = (side + 2 * QUIET_MODULES) * MODULE_PX
    canvas = np.full((img_side, img_side), 255, dtype=np.uint8)
    for row in range(side):
        y0 = (row + QUIET_MODULES) * MODULE_PX
        for col in range(side):
            if dark[row, col]:
                x0 = (col + QUIET_MODULES) * MODULE_PX
                canvas[y0:y0 + MODULE_PX, x0:x0 + MODULE_PX] = 0
    return canvas


def try_decode(canvas):
    try:
        from pyzbar import pyzbar
        from PIL import Image
    except ImportError:
        return None
    results = pyzbar.decode(Image.fromarray(canvas))
    if results:
        return results[0].data.decode(errors='replace')
    return None


def assemble_qrcode(is_hotdog):
    n = len(is_hotdog)
    side = int(round(n ** 0.5))
    if side * side != n:
        print(f'warning: {n} images is not a perfect square, cannot assemble a square grid')
        return

    qr_side = side // REDUNDANCY

    # orientation/polarity of the original grid isn't known up front, so try the
    # small set of combinations a photographic QR mosaic could have been laid out in
    for transposed in (False, True):
        for inverted in (False, True):
            canvas = build_canvas(is_hotdog, side, transposed, inverted)
            decoded = try_decode(canvas)
            if decoded is not None:
                iio.imwrite(QR_OUTPUT, canvas)
                print(f'assembled {side}x{side} images -> {qr_side}x{qr_side} QR modules '
                      f'-> {QR_OUTPUT} (transposed={transposed}, inverted={inverted})')
                print(f'FLAG: {decoded}')
                return

    # nothing decoded automatically: still save the most likely (untransposed,
    # non-inverted) version so it can be scanned/inspected by hand
    canvas = build_canvas(is_hotdog, side, transposed=False, inverted=False)
    iio.imwrite(QR_OUTPUT, canvas)
    print(f'assembled {side}x{side} images -> {qr_side}x{qr_side} QR modules -> {QR_OUTPUT}, '
          f'but no orientation/polarity decoded automatically '
          f'(pyzbar not installed, or image needs a manual look)')


if __name__ == '__main__':
    names, hotdog_flags = classify_all()
    assemble_qrcode(hotdog_flags)