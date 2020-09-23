from tkinter import *
from math import sqrt
from datetime import datetime
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision.transforms import ToTensor
from model import MNIST_CNN

w = 600
h = 600
brush_size = 30

# Load up the MNIST model.
model = torch.load("./MNIST_8x8_CNN.pth")

# In-memory PIL image drawn in parallel.
img = Image.new("L", (w, h), 255)
img_draw = ImageDraw.Draw(img)

def classify(img):
    """
    Given an 8x8 PIL image from the canvas, returns its predicted class label.
    """
    # Convert image to tensor - have to invert it
    img = ImageOps.invert(img)
    #img.save("drawing.png")
    tr = ToTensor()
    img_tensor = tr(img).unsqueeze(0)

    with torch.no_grad():
        model.train(False)
        model_out = model(img_tensor)

    return torch.argmax(model_out[0]).item()

def paint(event):
    """
    Called on mouse movement, draws rectangles on tkinter canvas and
    virtual PIL copy.
    """
    x = event.x
    y = event.y
    color='black'

    x1, y1 = (x-brush_size), (y-brush_size)
    x2, y2 = (x+brush_size), (y+brush_size)
    c.create_rectangle(x1,y1,x2,y2,fill=color,outline=color) # TKinter canvas
    img_draw.rectangle([(x1,y1), (x2,y2)], fill=color, outline=color) # PIL Parallel

def clf_on_click():
    """
    Classify Btn Click Handler.
    Runs classify, prints to screen.
    """
    lab = classify(img.resize((8,8)))
    lt.set(f"Classification:{str(lab)}")

# Set up tkinter components
root = Tk()
root.resizable(False, False)
root.title("PAINT")

c = Canvas(root, width=w, height=h, bg='white')
c.pack(expand=YES,fill=BOTH)
c.bind('<B1-Motion>', paint)

lt = StringVar()
message = Label(root, textvariable=lt)
message.pack(side=BOTTOM)

clf_btn = Button(root, text="Classify!", command=clf_on_click)
clf_btn.pack(side=BOTTOM)

root.mainloop()
