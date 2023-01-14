import glob
from PIL import Image
import numpy

 
files = glob.glob('png/*.png')

SIZE = 28
num = 0
for img in files:
    image = Image.open(img).convert('RGB')
    image = image.resize((SIZE, SIZE), Image.LANCZOS)
    img_color_resize = numpy.array(image)
    
    num = num + 1
    
    
    for color in range(0,3):
        out_file = "img_color_resize" + str(num) + "color"+ str(color + 1) + ".csv"
        numpy.savetxt(out_file, img_color_resize[:, :, color], delimiter=',', fmt="%.5f")