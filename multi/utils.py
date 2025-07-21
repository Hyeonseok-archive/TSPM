import numpy as np
from matplotlib import pyplot as plt, image as mpimg

# for grid plot for CU-BEMS
def plot_cubems_guide_map(path, title, path2=None, title2=None, mode='single', hs=12):

    img = mpimg.imread(path)

    if mode == 'single':
        plt.figure(figsize=(8, 8))
        plt.imshow(img)

        h, w = img.shape[:2]
        x = np.linspace(0, w, 21)
        y = np.linspace(0, h, hs+1)
        plt.xticks(x)
        plt.yticks(y)
        plt.tick_params(labelbottom=False, labelleft=False)

        plt.grid(color='r', linestyle='-', linewidth=1.0)
        plt.title(title)
        # plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        # plot img
        ax[0].imshow(img)

        h, w = img.shape[:2]
        x = np.linspace(0, w, 21)
        y = np.linspace(0, h, hs+1)
        ax[0].set_xticks(x)
        ax[0].set_yticks(y)
        ax[0].tick_params(labelbottom=False, labelleft=False)

        ax[0].grid(color='r', linestyle='-', linewidth=1.0)
        ax[0].set_title(title)

        # plot img2
        img2 = mpimg.imread(path2)
        ax[1].imshow(img2)

        h, w = img2.shape[:2]
        x = np.linspace(0, w, 21)
        y = np.linspace(0, h, hs+1)
        ax[1].set_xticks(x)
        ax[1].set_yticks(y)
        ax[1].tick_params(labelbottom=False, labelleft=False)

        ax[1].grid(color='r', linestyle='-', linewidth=1.0)
        ax[1].set_title(title2)

        # fig.suptitle('Plot Floors 1-2 & 3-7 guide map')
        # plt.tight_layout()
        plt.show()


# for make 2d dataset
def make_guidemap_list():
    # shape=(12*20)
    z0 = 0
    z1 = 'z1'
    z2 = 'z2'
    z3 = 'z3'
    z4 = 'z4'
    z5 = 'z5'

    # Floors 1-2
    row0 = [z2]*11 + [z1]*9
    row1 = row0

    row2 = [z3] + [z2]*9 + [z1]*9 + [z3]
    row3 = [z3] + [z2]*8 + [z1]*10 + [z3]
    row4 = row3

    row5 = [z3] + [z4]*4 + [z2]*4 + [z1]*10 + [z3]

    row6 = [z3] + [z4]*5 + [z3]*6 + [z1]*7 + [z3]
    row7 = row6

    row8 = [z3] + [z4]*5 + [z3]*5 + [z0] + [z1]*7 + [z3]
    row9 = row8

    row10 = [z0] + [z4]*5 + [z3]*5 + [z0] + [z1]*7 + [z0]
    row11 = row10

    guidemap_12 = [row0, row1, row2, row3, row4, row5,
                   row6, row7, row8, row9, row10, row11]

    # Floors 3-7
    row0 = [z0] + [z4]*7 + [z1]*11 + [z0]
    row1 = row0

    row2 = [z3] + [z4]*7 + [z1]*11 + [z3]
    row3 = row2

    row4 = [z3] + [z4]*9 + [z1]*9 + [z3]

    row5 = [z3] + [z5]*4 + [z4]*5 + [z1] + [z2]*8 + [z3]

    row6 = [z3] + [z5]*4 + [z3]*7 + [z2]*7 + [z3]
    row7 = row6

    row8 = [z3] + [z5]*4 + [z3]*6 + [z0] + [z2]*7 + [z3]
    row9 = row8

    row10 = [z0] + [z5]*4 + [z3]*6 + [z0] + [z2]*7 + [z0]
    row11 = row10

    guidemap_37 = [row0, row1, row2, row3, row4, row5,
                   row6, row7, row8, row9, row10, row11]

    return guidemap_12, guidemap_37