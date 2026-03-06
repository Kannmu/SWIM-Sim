import os
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

# Default Input Directory (can be overridden)
INPUT_DIR = r'k:\Work\SWIM\Sim\Outputs_Experiment1\Python_Plots'
OUTPUT_DIR = r'D:\Data\OneDrive\Papers\SWIM\Image\SI'

# Output Filenames
OUTPUT_FILENAME_SPATIAL = 'Figure_Sim_1_Full_Spatial.png'
OUTPUT_FILENAME_TEMPO = 'Figure_Sim_1_Full_Tempo.png'

# Target Size
MAX_EDGE_PIXELS = 4096

# Row Definitions (Modulation Types)
# Defines the order of rows in the combined figure
MODULATIONS = ['DLM_2', 'DLM_3', 'LM_C', 'LM_L', 'ULM_L']

# Column Definitions (Image Filename Patterns)
# Spatial Figure Columns (5 rows x 4 columns)
SPATIAL_COLUMNS = [
    '{mod}_1_energy_rms.png',
    '{mod}_2_peak_stress.png',
    '{mod}_3_rate_of_change.png',
    '{mod}_4_gradient_snapshot.png'
]

# Tempo Figure Columns (5 rows x 4 columns)
TEMPO_COLUMNS = [
    'comparison_XT_diagram_{mod}_tau_xz.png',
    '{mod}_5_waveform.png',
    '{mod}_snapshots_tau_xy.png',
    '{mod}_snapshots_uz.png'
]

# =============================================================================
# Functions
# =============================================================================

def combine_images(modulations, column_patterns, output_filename, input_dir=INPUT_DIR):
    """
    Combines images into a grid based on modulations (rows) and patterns (columns).
    
    Args:
        modulations (list): List of modulation names (rows).
        column_patterns (list): List of filename patterns with {mod} placeholder (columns).
        output_filename (str): Name of the output file.
        input_dir (str): Directory containing source images.
    """
    rows = len(modulations)
    cols = len(column_patterns)
    
    print(f"Combining {rows}x{cols} grid for {output_filename}...")
    
    # 1. Load images and determine grid cell size
    grid_images = [[None for _ in range(cols)] for _ in range(rows)]
    max_w = 0
    max_h = 0
    
    valid_images_count = 0
    
    for r, mod in enumerate(modulations):
        for c, pattern in enumerate(column_patterns):
            filename = pattern.format(mod=mod)
            filepath = os.path.join(input_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath)
                    grid_images[r][c] = img
                    max_w = max(max_w, img.width)
                    max_h = max(max_h, img.height)
                    valid_images_count += 1
                except Exception as e:
                    print(f"  [Error] Could not open {filename}: {e}")
            else:
                print(f"  [Warning] File not found: {filename}")

    if valid_images_count == 0:
        print("  [Error] No valid images found. Skipping.")
        return

    print(f"  Max cell size: {max_w}x{max_h} pixels")

    # 2. Create canvas
    full_width = max_w * cols
    full_height = max_h * rows
    
    # Create white background
    combined_image = Image.new('RGB', (full_width, full_height), (255, 255, 255))
    
    # 3. Paste images
    for r in range(rows):
        for c in range(cols):
            img = grid_images[r][c]
            if img:
                # Resize if necessary to match the grid cell size (using high-quality resampling)
                if img.width != max_w or img.height != max_h:
                    img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
                
                x = c * max_w
                y = r * max_h
                combined_image.paste(img, (x, y))
    
    # 4. Resize final image to target max edge
    scale_factor = 1.0
    if full_width > full_height:
        if full_width > MAX_EDGE_PIXELS:
            scale_factor = MAX_EDGE_PIXELS / full_width
    else:
        if full_height > MAX_EDGE_PIXELS:
            scale_factor = MAX_EDGE_PIXELS / full_height
            
    if scale_factor < 1.0: # Only downscale, never upscale
        new_size = (int(full_width * scale_factor), int(full_height * scale_factor))
        print(f"  Resizing result from {full_width}x{full_height} to {new_size} (scale={scale_factor:.4f})...")
        combined_image = combined_image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        print(f"  Final size: {full_width}x{full_height} (No resizing needed)")
    
    # 5. Save
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        combined_image.save(output_path)
        print(f"  [Success] Saved to: {output_path}")
    except Exception as e:
        print(f"  [Error] Could not save image: {e}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return

    # Generate Spatial Figure
    combine_images(MODULATIONS, SPATIAL_COLUMNS, OUTPUT_FILENAME_SPATIAL)
    
    print("-" * 40)
    
    # Generate Tempo Figure
    combine_images(MODULATIONS, TEMPO_COLUMNS, OUTPUT_FILENAME_TEMPO)

if __name__ == "__main__":
    main()