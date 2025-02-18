from tkinter import mainloop

from applications.LevelDrawer import LevelDrawer


level_path = '../train_datasets/single_structure'
generated_levels = '../generated_levels/main_set/'

if __name__ == '__main__':
    level_drawer = LevelDrawer(
        level_path = generated_levels,
        drawing_canvas_size = (60, 60),
        science_birds_path ='resources/science_birds/win-new/',
    )
    mainloop()
