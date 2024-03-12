from glob import glob
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# collect files
def collect_files(img_ind, param="proj_out"):

    files = glob(f'/results/counterfactuals/class_grad_eval/grad_{img_ind}_*_{param}.npy')

    if param == "classifier_score":
        files = [f for f in files if 'implicit' not in f]
    # sort files by number
    files.sort(key=lambda x: int(x.split("_")[4]))

    files = files[::-1]

    print([int(f.split("_")[4]) for f in files])

    return files

def create_gif(img_id=0, param='proj_out', same_scale=True):

    files = collect_files(img_id, param=param)
    print(f'Number of files: {len(files)}')

    # visualize in a gif
    Figure = plt.figure()
    ax = plt.axes(projection='3d')

    frame_data = np.load(files[0])[img_id, 0]
    print(frame_data.shape)
    x, y = np.meshgrid(np.arange(frame_data.shape[0]), np.arange(frame_data.shape[1]))
    z = frame_data
    surface = ax.plot_surface(x, y, z, cmap='cool', alpha=0.6)
    
    if param == "implicit_classifier_score" or same_scale:
        ax.set_zlim(-0.1, 0.1)
    else:
        ax.set_zlim(-0.015, 0.015)

    time_text = plt.title(f"Frame: 0")

    def update(frame, surface, time_text): 

        frame_file = files[frame]

        frame_data = np.load(frame_file)[img_id, 0]

        ax.clear()
        # surface.remove()
        surface = ax.plot_surface(x, y, frame_data, cmap='cool', alpha=0.6)

        if param == "implicit_classifier_score" or same_scale:
            ax.set_zlim(-0.1, 0.1)
        else:
            ax.set_zlim(-0.015, 0.015)

        ax.set_title(f"Frame: {frame}")
    
        # line is set with new values of x and y
        # surface.set_verts(frame_data)
        return surface, [time_text]


    anim_created = FuncAnimation(fig=Figure, func=update, fargs=(surface, time_text), frames=191, interval=120)

    # video = anim_created.to_html5_video()
    # html = display.HTML(video)
    # display.display(html) 

    anim_created.save(filename=f"/results/counterfactuals/class_grad_eval/{img_id}_{param}.gif", writer="pillow")

    # good practice to close the plt object.
    plt.close()

def main():

    for img_id in range(4):
        for param in ['proj_out', 'classifier_score', 'implicit_classifier_score', 'grad_classifier', 'interpolated_out']:

            create_gif(img_id=img_id, param=param)

if __name__ == '__main__':
    main()
