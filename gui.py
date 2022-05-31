import tkinter as tk
from tkinter import Toplevel
from tkinter import font as tkFont
from tkinter.messagebox import showinfo
from tkinter.filedialog import askdirectory
from PIL import ImageTk, Image
from generate import main_flow
from glob import glob

model_list = ['yellow', 'western', 'western_old']


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.pack()
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        fontStyle = tkFont.Font(size=20)
        choselabel = tk.Label(self, text="選擇人種：", font=fontStyle)
        choselabel.grid(row=1, column=0, padx=10, pady=5, sticky=tk.N+tk.W)
        self.listbox = tk.Listbox(self, font=fontStyle)
        self.listbox["height"] = 3
        self.listbox.insert(1, "亞洲人")
        self.listbox.insert(2, "歐美人")
        self.listbox.insert(3, "歐美人_舊")
        self.listbox.select_set(0)
        self.listbox.grid(row=1, column=1, padx=10, pady=5, sticky=tk.N+tk.W)
        self.listbox.select_set(0)
        geneeate_num_label = tk.Label(self, text="生成數量：", font=fontStyle)
        geneeate_num_label.grid(row=2, column=0, padx=10,
                                pady=5, sticky=tk.N+tk.W)
        vcmd = (self.register(validate), '%P')
        self.geneeate_num_input = tk.Entry(
            self, width=20, font=fontStyle, validate='key', validatecommand=vcmd)
        self.geneeate_num_input.grid(
            row=2, column=1, padx=10, pady=5, sticky=tk.S)

        save_path_label = tk.Label(self, text="儲存路徑：", font=fontStyle)
        save_path_label.grid(row=3, column=0, padx=10,
                             pady=5, sticky=tk.N+tk.W)
        self.save_path_input = tk.Entry(self, width=20, font=fontStyle)
        self.save_path_input.grid(
            row=3, column=1, padx=10, pady=5, sticky=tk.S)
        self.button = tk.Button(
            self, text="瀏覽", font=fontStyle, bg="WHITE")
        self.button.grid(
            row=5, column=0, padx=10, pady=5, sticky=tk.S)
        self.button["command"] = self.select_dir
        generate_commit = tk.Button(
            self, text="生成", font=fontStyle, bg="WHITE")
        generate_commit["command"] = self.commit_event
        generate_commit.grid(row=5, column=1, padx=10, pady=5,
                             sticky=tk.N+tk.E+tk.S+tk.W)
        # self.button = tk.Button(
        #     self, text="瀏覽", font=fontStyle, bg="WHITE")
        # self.button["command"] = self.brower_event
        # self.button.grid(row=6, column=0, padx=10, pady=5,
        #                  sticky=tk.N+tk.E+tk.S+tk.W)

    def commit_event(self):
        ckpt = None
        # ckpt = f"checkpoint/{'western' if self.listbox.curselection()[0] else 'yellow'}.pt"
        model_name = model_list[self.listbox.curselection()[0]]
        print(model_name)
        ckpt = f"checkpoint/{model_name}.pt"
        save_path = self.save_path_input.get(
        ) if self.save_path_input.get() != '' else '/tmp/generate'
        print(save_path)
        pics = self.geneeate_num_input.get() if self.save_path_input.get() != '' else 1000

        main_flow(ckpt, int(pics), save_path)
        showbox()

    def select_dir(self):
        selectPath()
        self.save_path_input.insert(0, path.get())

    def brower_event(self):
        selectPath()
        win = Toplevel(self)
        fontStyle = tkFont.Font(size=20)
        win.title('brower files')
        win.geometry("500x350+200+20")
        win.grid()
        i = 0
        index.set(i)
        img_list = glob(f'{path.get()}/*.*g')
        img = ImageTk.PhotoImage(Image.open(img_list[i]))
        img_label = tk.Label(win, height=256, width=256,
                             bg='gray94', fg='blue', image=img)
        img_label.grid(row=0, column=1, sticky=tk.N+tk.E+tk.S+tk.W)
        left_button = tk.Button(
            win, text='left', font=fontStyle, command=lambda: change_img('-', img_label, img_list))
        left_button.grid(row=1, column=0, padx=10, pady=5,
                         sticky=tk.N+tk.E+tk.S+tk.W)
        right_button = tk.Button(
            win, text='right', font=fontStyle, command=lambda: change_img('+', img_label, img_list))
        right_button.grid(row=1, column=3, padx=10, pady=5,
                          sticky=tk.N+tk.E+tk.S+tk.W)
        win.mainloop()


def change_img(op, img_label, img_list):
    if op == '+':
        i = index.get()+1
        index.set(min(len(img_list)-1, i))
    else:
        i = index.get()-1
        index.set(max(0, i))
    img = ImageTk.PhotoImage(Image.open(img_list[index.get()]))
    img_label.imgtk = img
    img_label.config(image=img)


def selectPath():
    path_ = askdirectory()
    path.set(path_)


def showbox():
    showinfo('finish generate', '已經完成')


def validate(P):
    if str.isdigit(P) or P == '':
        return True
    else:
        return False


root = tk.Tk()
path = tk.StringVar()
index = tk.IntVar()
root.title("image generate")
root.geometry("500x300+200+20")
app = Application(root)
root.mainloop()
