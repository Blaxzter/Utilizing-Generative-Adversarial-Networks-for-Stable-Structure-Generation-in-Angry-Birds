from tkinter import mainloop

from applications.LevelDrawer import LevelDrawer

if __name__ == '__main__':
    level_drawer = LevelDrawer(
        level_path = '../train_datasets/single_structure',
        drawing_canvas_size = (60, 60),
        science_birds_path = './resources/science_birds/win-slow/',
    )
    mainloop()
