import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def upscale_and_resize(input_folder, output_folder, target_size=(224, 224), model_name='RealESRGAN_x4plus'):
    """
    Upscale images using RealESRGAN and resize them to a fixed size, preserving folder structure.
    
    Args:
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save processed images.
        target_size (tuple): Final size (width, height) of the output images.
        model_name (str): Model name for RealESRGAN.
    """

    # Determine model parameters
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load model weights
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(url=url, model_dir='weights', progress=True)

    # Initialize the RealESRGAN upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )

    skipped = 0

    # Process images recursively
    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            imgname, extension = os.path.splitext(file)

            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Skipping {input_path}, unable to read image.")
                continue

            try:
                # Get image dimensions
                height, width = img.shape[:2]

                # Skip if either dimension is smaller than 112
                if width < 112 or height < 112:
                    skipped += 1
                    print(f"Skipping {input_path}, dimensions too small ({width}x{height}). Skipped {skipped} images.")
                    continue

                # Determine scale factor
                if width < target_size[0] or height < target_size[1]:
                    scale = 2  # Use a fixed scale of 2 if smaller than target
                else:
                    scale = 1  # No upscaling needed

                if scale > 1:
                    # Upscale the image if necessary
                    output, _ = upsampler.enhance(img, outscale=scale)
                else:
                    # Use the original image if no upscaling is needed
                    output = img

                # Resize to the target size
                resized = cv2.resize(output, target_size, interpolation=cv2.INTER_AREA)

                # Save the output image
                save_path = os.path.join(output_dir, f'{imgname}_resized.png')
                cv2.imwrite(save_path, resized)
            except RuntimeError as error:
                print(f'Error processing {input_path}: {error}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input folder containing images')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder to save processed images')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help='Model name for RealESRGAN')
    parser.add_argument('--width', type=int, default=224, help='Target width for the final image')
    parser.add_argument('--height', type=int, default=224, help='Target height for the final image')
    args = parser.parse_args()

    upscale_and_resize(
        input_folder=args.input,
        output_folder=args.output,
        target_size=(args.width, args.height),
        model_name=args.model_name
    )
