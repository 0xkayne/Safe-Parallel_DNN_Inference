from PIL import Image
import glob
import math
import os

def stitch_images(image_dir, output_filename, cols=3):
    """
    Stitches all *_chart.png files in image_dir into a single grid image.
    """
    # Find all chart images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*_chart.png")))
    if not image_paths:
        print(f"No *_chart.png images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images in {image_dir}")

    # Load all images
    images = [Image.open(f) for f in image_paths]
    
    # Get dimensions (assume all charts are roughly same size/aspect ratio)
    # We use the size of the first image as reference
    w, h = images[0].size
    
    n_images = len(images)
    rows = math.ceil(n_images / cols)
    
    # Create blank white canvas
    canvas_w = w * cols
    canvas_h = h * rows
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    
    # Paste images
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        
        # Resize if necessary (though our charts should be uniform)
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            
        canvas.paste(img, (c * w, r * h))
        
    # Save output
    canvas.save(output_filename)
    print(f"[OK] Saved combined chart to: {output_filename}")

def main():
    print("Combining charts...")
    
    # Combine Server Charts
    if os.path.exists('server-chart'):
        stitch_images('server-chart', 'combined_server_charts_grid.png', cols=2)
        
    # Combine Network Charts
    if os.path.exists('network-chart'):
        stitch_images('network-chart', 'combined_network_charts_grid.png', cols=2)

if __name__ == "__main__":
    main()
