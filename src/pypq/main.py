# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from src.cabaret import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def source1():
    return 1.

def ic1(x, y):
    center = [0.5, 0.5]
    std = 0.15
    return np.exp((-1. / (2. * std ** 2)) * ((x - center[0]) ** 2 + (y - center[1]) ** 2))

def exact1(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    nx = 32
    ny = nx
    x_left = 0.
    x_right = 1.
    y_bottom = 0.
    y_top = 1.

    cabaret = CABARET(nx, ny, y_bottom, y_top, x_left, x_right, index='std')

    cabaret.boundary_conditions(['Periodic', 'Dirichlet', 'Periodic', 'Dirichlet'])
    print(cabaret.m_nx)
    print(cabaret.m_ny)
    print(cabaret.m_nx_adv)
    print(cabaret.m_ny_adv)

    ''' Initial conditions '''
    cabaret.set_initial_conditions(ic1)

    print(cabaret.m_cons.shape)
    print(cabaret.m_adv_x.shape)
    print(cabaret.m_adv_y.shape)

    # labels = grid.labels
    # print(labels)
    # print(grid.forcing_vector)
    # solution = np.linalg.inv(grid.laplacian_matrix) @ grid.forcing_vector
    # solution, exit_code = sp.sparse.linalg.gmres(grid.laplacian_matrix, grid.forcing_vector)
    # print(exit_code)

    x_coords = cabaret.get_x_coords
    y_coords = cabaret.get_y_coords
    solutions = cabaret.solutions

    X, Y = np.meshgrid(x_coords, y_coords)

    z_min = np.min(solutions)
    z_max = np.max(solutions)
    solution = solutions.reshape(ny, nx)

    # Customise your visuals here
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica"
    # })
    fig = plt.figure(figsize=(10, 5))

    # Subplot 1: imshow
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(solution, extent=[x_left, x_right, y_bottom, y_top], origin='lower', cmap='viridis', aspect='auto',
                    vmin=z_min, vmax=z_max)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_xlim([x_left, x_right])
    ax1.set_ylim([y_bottom, y_top])
    cbar = fig.colorbar(im, ax=ax1, label=r'$u$')  # Individual color bar for imshow

    # Subplot 2: plot_surface with adjustable camera angle
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, solution, cmap='Blues_r', edgecolor='none')

    # Adjust the camera angle (elevation, azimuth)
    ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_zlabel(r'$u$')
    ax2.set_xlim([x_right, x_left])
    ax2.set_ylim([y_top, y_bottom])
    ax2.set_xlim([x_right, x_left])
    ax2.set_zlim([z_min, z_max])
    plt.suptitle(r'Solutions', fontsize=16, usetex=True)

    # Show the plot
    plt.tight_layout()  # rect=[0, 0, 1, 0.95]
    plt.show()

    # Display the grid
    # fig, ax = plt.subplots()
    # ax.matshow(labels, cmap='coolwarm')
    # for i in range(ny):
    #     for j in range(nx):
    #         ax.text(j, i, str(labels[i, j]), va='center', ha='center', color='black')
    # plt.title("Onion numbering system")
    # plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
