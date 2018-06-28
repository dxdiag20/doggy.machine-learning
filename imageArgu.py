from PIL import Image
image = Image.open("/Users/hangeulbae/Desktop/test.jpeg")
newImage = image.resize((300, 300))
newImage.save("/Users/hangeulbae/Desktop/test2.png")
