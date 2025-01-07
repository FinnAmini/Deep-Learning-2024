from realesrgan import RealESRGAN

from PIL import Image

# Load image
image = Image.open('input.jpg')

# Initialize the model
model = RealESRGAN(device='cuda')  # Use 'cpu' if no GPU available
model.load_weights('weights/RealESRGAN_x4.pth')

# Upscale image
upscaled_image = model.predict(image)

# Save the result
upscaled_image.save('output_upscaled.jpg')