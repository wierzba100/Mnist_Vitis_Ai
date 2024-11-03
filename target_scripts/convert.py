from PIL import Image
import numpy as np

img = Image.open('image.png')
print(img.size)

img = img.convert('L')

img = img.resize((28,28))

img_array = np.array(img)
img_array = img_array.reshape(28, 28, 1)

print(img_array.shape)

np.save('output_array.npy', img_array)
