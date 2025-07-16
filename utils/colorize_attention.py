import matplotlib

def colorize(words, color_array):
    """Generate a colored HTML string for visualization, where word background intensity reflects its importance."""
    cmap=matplotlib.cm.Blues
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        # Maps each attention score (or weight) to a color using matplotlib.cm.Blues colormap.
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        # print(color)
        
        # Wraps each word with HTML <span> using background color:
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

