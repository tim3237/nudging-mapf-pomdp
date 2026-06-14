# created with gpt-4o-mini-2024-07-18
import numpy as np
from .simulator import Simulator
try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    matplotlib_ready = True
except ModuleNotFoundError as _:
    matplotlib_ready = False


def simulation_to_gif(simulator: Simulator, filename="rollout.gif"):
    if not matplotlib_ready:
        raise ImportError("Cannot setup gif without matplotlib. Install matplotlib first.")
    simulator.state.clear_allocation()
    background = simulator.state.geometry.geometry[0].to("cpu")
    agents = simulator.state.history.get_agent_position_list()

    # Define colors for the background values
    color_map = {
        -1: 'black',
        0: 'white',
        2: 'red',
        3: 'purple',
        4: 'orange',
        5: 'yellow',
        6: 'magenta',
        7: 'cyan',
        8: 'teal',
        9: 'gray',
        10: 'olive',
        11: 'brown',
        12: 'tan'
    }

    # Create a function to create a color array from the tensor
    def create_color_array(tensor):
        color_array = np.empty(tensor.shape + (3,), dtype=float)  # RGB color array
        for value, color in color_map.items():
            color_array[tensor == value] = mcolors.to_rgb(color)
        return color_array

    # Create a color array from the background tensor
    background_colors = create_color_array(background)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    if len(background_colors.shape) == 4:
        im = ax.imshow(background_colors[0])
    else:
        im = ax.imshow(background_colors)

    # Add a text object to display the frame number
    frame_text = ax.text(0.5, 1.05, f'Frame: 0', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, bbox=dict(facecolor='white', alpha=0.5))

    # Function to update the frame
    def update(frame):
        # Reset the background
        if len(background_colors.shape) == 4:
            colors = background_colors[frame].copy()
        else:
            colors = background_colors.copy()

        # Highlight the specific tiles based on the highlight tensor
        if frame < len(agents):
            # Get the indices of the highlighted tiles in the highlight tensor
            highlighted_indices = agents[frame]
            for x, y, c in highlighted_indices:
                if c == 1:  # only main goal
                    colors[x, y, :] = mcolors.to_rgb('blue')  # Highlight color
                else:
                    colors[x, y, :] = mcolors.to_rgb('green')  # Highlight color

        # Update the frame number text
        frame_text.set_text(f'Frame: {frame + 1}')  # Frame numbers start from 1

        # Update the image with the new color array
        im.set_array(colors)
        return im, frame_text

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(agents), interval=1000//25, blit=False)
    ani.save(filename, writer="pillow", fps=50)
