from PIL import Image

def resize_image(input_path, output_path, size=(1600, 1600)):
    with Image.open(input_path) as image:
        # Resize the image
        resized_image = image.resize(size, Image.LANCZOS)
        
        # Save the resized image to the output path
        resized_image.save(output_path)

# Usage example:


resize_image('profile.png', 'profile.png')
