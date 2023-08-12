from ete3 import Tree, TreeStyle, TextFace

t = Tree()
A = t.add_child(name = "A")
B = t.add_child(name = "B")
C = A.add_child(name = "C")
D = C.add_sister(name = "D")

R = A.add_child(name = "R")


# Add two text faces to different columns
R.add_face(TextFace("hola "), column=1, position = "branch-right")
A.add_face(TextFace("mundo!"), column=0, position = "branch-right")


ts = TreeStyle()
ts.show_scale = False
ts.show_leaf_name = True

t.show(tree_style = ts)

# t = Tree()
# t.populate(30)
# ts = TreeStyle()
#
# ts.show_leaf_name = True
# ts.mode = "c"
# ts.arc_start = -180 # 0 degrees = 3 o'clock
# ts.arc_span = 180
#
# t.show(tree_style=ts)
