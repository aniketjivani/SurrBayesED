# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


plt.rc("axes.spines", right=True, top=True)
plt.rc("figure",
       dpi=300,
      #  figsize=(9, 3)
      )
plt.rc("font", family="serif")
plt.rc("legend", edgecolor="none", frameon=True)
plt.style.use("dark_background")

def draw_node(ax, x, y, radius=0.5):
    circle = Circle((x, y), radius, edgecolor="white", facecolor="none", linewidth=2.5)
    ax.add_patch(circle)

def connect_nodes(ax, x1, y1, x2, y2, node_radius=0.5):
    dx, dy = x2 - x1, y2 - y1
    distance = (dx**2 + dy**2)**0.5
    offset_x = dx / distance * node_radius
    offset_y = dy / distance * node_radius
    line = Line2D([x1 + offset_x, x2 - offset_x], [y1 + offset_y, y2 - offset_y], color="white", linewidth=2.5)
    ax.add_line(line)


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")

input_layer_x = 2
hidden_layer_x = 5
output_layer_x = 8

input_layer_y = [4.5, 3, 1.5]  # y-coordinates of input layer nodes
hidden_layer_y = [5, 4, 2, 1]  # y-coordinates of hidden layer nodes
output_layer_y = [4, 2]         # y-coordinates of output layer nodes

for y1 in input_layer_y:
    for y2 in hidden_layer_y:
        connect_nodes(ax, input_layer_x, y1, hidden_layer_x, y2)

for y1 in hidden_layer_y:
    for y2 in output_layer_y:
        connect_nodes(ax, hidden_layer_x, y1, output_layer_x, y2)

for y in input_layer_y:
    draw_node(ax, input_layer_x, y)

for y in hidden_layer_y:
    draw_node(ax, hidden_layer_x, y)

for y in output_layer_y:
    draw_node(ax, output_layer_x, y)

plt.show()
# %%