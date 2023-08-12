#!/usr/bin/python

# SOURCE https://stackoverflow.com/questions/35139155/tkinter-notebook-too-many-tabs-for-window-width
# NO STELEROONI

# Try to work with older version of Python
from __future__ import print_function

import sys

if sys.version_info.major < 3:
    import Tkinter as tk
    import Tkinter.ttk as ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk

#============================================================================
#   MAIN CLASS
class Main(tk.Frame):
    """ Main processing
    """
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)

        self.root = root
        self.root_f = tk.Frame(self.root)

        self.width = 700
        self.height = 300

        # Create a canvas and scroll bar so the notebook can be scrolled
        self.nb_canvas = tk.Canvas(self.root_f, width=self.width, height=self.height)
        self.nb_scrollbar = tk.Scrollbar(self.root_f, orient='horizontal')

        # Configure the canvas and scrollbar to each other
        self.nb_canvas.config(yscrollcommand=self.nb_scrollbar.set,
                              scrollregion=self.nb_canvas.bbox('all'))
        self.nb_scrollbar.config(command=self.nb_canvas.xview)

        # Create the frame for the canvas window, and place
        self.nb_canvas_window = tk.Frame(self.nb_canvas, width=self.width, height=self.height)
        self.nb_canvas.create_window(0, 0, window=self.nb_canvas_window)

        # Put the whole notebook in the canvas window
        self.nb = ttk.Notebook(self.nb_canvas_window)

        self.root_f.grid()
        self.nb_canvas.grid()
        self.nb_canvas_window.grid()
        self.nb.grid(row=0, column=0)
        self.nb_scrollbar.grid(row=1, column=0, sticky='we')

        self.nb.enable_traversal()

        for count in range(20):
            self.text = 'Lots of text for a wide Tab ' + str(count)
            self.tab = tk.Frame(self.nb)
            self.nb.add(self.tab, text=self.text)
            # Create the canvas and scroll bar for the tab contents
            self.tab_canvas = tk.Canvas(self.tab, width=self.width, height=self.height)
            self.tab_scrollbar = tk.Scrollbar(self.tab, orient='vertical')
            # Convigure the two together
            self.tab_canvas.config(xscrollcommand=self.tab_scrollbar.set,
                                      scrollregion=self.tab_canvas.bbox('all'))
            self.tab_scrollbar.config(command=self.tab_canvas.yview)
                # Create the frame for the canvas window
            self.tab_canvas_window = tk.Frame(self.tab_canvas)
            self.tab_canvas.create_window(0, 0, window=self.tab_canvas_window)

            # Grid the content and scrollbar
            self.tab_canvas.grid(row=1, column=0)
            self.tab_canvas_window.grid()
            self.tab_scrollbar.grid(row=1, column=1, sticky='ns')

            # Put stuff in the tab
            for count in range(20):
                self.text = 'Line ' + str(count)
                self.line = tk.Label(self.tab_canvas_window, text=self.text)
                self.line.grid(row=count, column=0)

        self.root.geometry('{}x{}+{}+{}'.format(self.width, self.height, 100, 100))

        return

#   MAIN (MAIN) =======================================================
def main():
    """ Run the app
    """
    # # Create the screen instance and name it
    root = tk.Tk()
    # # This wll control the running of the app.
    app = Main(root)
    # # Run the mainloop() method of the screen object root.
    root.mainloop()
    root.quit()

#   MAIN (STARTUP) ====================================================
#   This next line runs the app as a standalone app
if __name__ == '__main__':
    # Run the function name main()
    main()