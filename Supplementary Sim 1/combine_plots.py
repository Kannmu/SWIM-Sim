import os
from PIL import Image

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'Outputs_Supplementary_Experiment1', 'Python_Plots')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Image', 'SI')
OUTPUT_FILENAME_SPATIAL = 'Figure_Supplementary_Sim_1_Full_Spatial.png'
OUTPUT_FILENAME_TEMPO = 'Figure_Supplementary_Sim_1_Full_Tempo.png'
MAX_EDGE_PIXELS = 4096

METHODS = ['ulm_l_m0p2', 'ulm_l_m0p5', 'ulm_l_m1p0', 'ulm_l_m1p2', 'ulm_l_m2p0']
SPATIAL_COLUMNS = [
    '{method}_1_energy_rms.png',
    '{method}_2_peak_stress.png',
    '{method}_3_rate_of_change.png',
    '{method}_4_gradient_snapshot.png',
]
TEMPO_COLUMNS = [
    '{method}_xt_diagram_tau_xz.png',
    '{method}_5_waveform.png',
    '{method}_snapshots_tau_xy.png',
    '{method}_snapshots_uz.png',
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def combine_images(methods, column_patterns, output_filename, input_dir=INPUT_DIR):
    rows = len(methods)
    cols = len(column_patterns)

    print(f'Combining {rows}x{cols} grid for {output_filename}...')

    grid_images = [[None for _ in range(cols)] for _ in range(rows)]
    max_w = 0
    max_h = 0
    valid_images_count = 0

    for r, method in enumerate(methods):
        for c, pattern in enumerate(column_patterns):
            filename = pattern.format(method=method)
            filepath = os.path.join(input_dir, filename)

            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath)
                    grid_images[r][c] = img
                    max_w = max(max_w, img.width)
                    max_h = max(max_h, img.height)
                    valid_images_count += 1
                except Exception as e:
                    print(f'  [Error] Could not open {filename}: {e}')
            else:
                print(f'  [Warning] File not found: {filename}')

    if valid_images_count == 0:
        print('  [Error] No valid images found. Skipping.')
        return

    print(f'  Max cell size: {max_w}x{max_h} pixels')

    full_width = max_w * cols
    full_height = max_h * rows
    combined_image = Image.new('RGB', (full_width, full_height), (255, 255, 255))

    for r in range(rows):
        for c in range(cols):
            img = grid_images[r][c]
            if img:
                if img.width != max_w or img.height != max_h:
                    img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)

                x = c * max_w
                y = r * max_h
                combined_image.paste(img, (x, y))

    scale_factor = 1.0
    if full_width > full_height:
        if full_width > MAX_EDGE_PIXELS:
            scale_factor = MAX_EDGE_PIXELS / full_width
    else:
        if full_height > MAX_EDGE_PIXELS:
            scale_factor = MAX_EDGE_PIXELS / full_height

    if scale_factor < 1.0:
        new_size = (int(full_width * scale_factor), int(full_height * scale_factor))
        print(f'  Resizing result from {full_width}x{full_height} to {new_size} (scale={scale_factor:.4f})...')
        combined_image = combined_image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        print(f'  Final size: {full_width}x{full_height} (No resizing needed)')

    ensure_dir(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        combined_image.save(output_path)
        print(f'  [Success] Saved to: {output_path}')
    except Exception as e:
        print(f'  [Error] Could not save image: {e}')



def main():
    if not os.path.exists(INPUT_DIR):
        print(f'Error: Input directory does not exist: {INPUT_DIR}')
        return

    combine_images(METHODS, SPATIAL_COLUMNS, OUTPUT_FILENAME_SPATIAL)
    print('-' * 40)
    combine_images(METHODS, TEMPO_COLUMNS, OUTPUT_FILENAME_TEMPO)


if __name__ == '__main__':
    main()
