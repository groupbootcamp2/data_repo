import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd, NW
from tkinter.messagebox import showinfo
from PIL import ImageTk,Image
import crop_img

root = tk.Tk()
root.title('upload pictures')
root.resizable(False, False)
root.geometry('1000x700')

def select_file():
    filetypes = (
        ('text files', '*.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
    )
    image=crop_img.get_crop_img(str(filename))  #get the crop img

    #try to check and show the crop img

    # print(type(image))
    # test = ImageTk.PhotoImage(image)
    #
    # label1 = tk.Label(image=test)
    # label1.image = test
    #
    # # Position image
    # label1.place(x= 200, y = 200)


# open button
open_button = ttk.Button(
    root,
    text='Open a File',
    command=select_file
)

open_button.pack(expand=True)

root.mainloop()# run the application