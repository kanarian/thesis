from PIL import Image

def plot_grid(images, n_rows, n_cols):
    width, height = images[0].size
    canvas_width = n_cols * width
    canvas_height = n_rows * height

    canvas = Image.new('L', (canvas_width, canvas_height))
    for i, image in enumerate(images):
        x = (i % n_cols) * width
        y = (i // n_cols) * height
        canvas.paste(image, (x, y))
    canvas.show()