from PIL import Image
im = Image.open('test.png','r').convert('RGB')
#im.show()
width, height = im.size
print(width)
print(height)
nimg = Image.new('RGB',(width,height))
pixels = nimg.load()

for i in range(width):
	for j in range(height):
		r, g, b = im.getpixel((i, j))
		r=r+10
		g=g+20
		b=b-30
		pixels[i,j] = (int(r),int(g),int(b))
nimg.show()		
