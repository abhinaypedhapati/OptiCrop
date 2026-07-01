import numpy as np
from PIL import Image

image_path = 'C:/Users/pedha/.gemini/antigravity-ide/brain/a941cd37-8820-4a61-a122-d2cf9910e499/media__1782933398316.png'
img = Image.open(image_path).convert('L')
arr = np.array(img)

# Print row-wise and column-wise averages to see the spacing peaks
row_avg = np.mean(arr, axis=1)
col_avg = np.mean(arr, axis=0)

print("Row averages length:", len(row_avg))
print("Col averages length:", len(col_avg))

# Find the separator rows (local maxima of brightness close to 255)
# In our 682x1024 image, let's find the rows that are very white
white_rows = np.where(row_avg > 250)[0]
white_cols = np.where(col_avg > 250)[0]

print("White rows indices:", white_rows)
print("White cols indices:", white_cols)
