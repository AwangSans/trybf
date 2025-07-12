import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


def load_and_plot(fk_file, params):
    # Load fk from .npy file
    fk = np.load(fk_file)

    fk = 10*np.log10(fk/fk.max())
    # Extract parameters
    smin = params['smin']
    smax = params['smax']
    cap_find = params['cap_find']
    cap_fave = params['cap_fave']
    nsamp = params['nsamp']
    dt = params['dt']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(fk.T, extent=[smin, smax, smax, smin], cmap='gist_stern_r', interpolation='none')
    plt.title('Slowness Spectrum at %.03f +- %.03f[Hz]' % (cap_find / (nsamp * dt), cap_fave / (nsamp * dt)))
    ax.set_xlim([smin, smax])
    ax.set_ylim([smin, smax])
    ax.set_xlabel('East/West Slowness [s/deg]')
    ax.set_ylabel('North/South Slowness [s/deg]')

    circle = plt.Circle((0, 0), np.lib.scimath.sqrt((0.3 * 111.19)**2), color='w', fill=False, alpha=0.4)
    plt.gcf().gca().add_artist(circle)

    circle = plt.Circle((0, 0), np.lib.scimath.sqrt((0.24 * 111.19)**2), color='w', fill=False, alpha=0.4)
    plt.gcf().gca().add_artist(circle)

    cbar = fig.colorbar(im)
    cbar.set_label('absolute power', rotation=270)

    # Save the image instead of showing it
    img_name = os.path.basename(fk_file).split('.')[0] + '.png'
    img_path = os.path.join('./fk_fig_test', img_name)
    print(img_path)
    plt.savefig(img_path)
    plt.close()


def process_directory(directory, params):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f"Error: {directory} does not exist or is not a directory.")
        return

    save_dir = './fk_fig_test'
    os.makedirs(save_dir,exist_ok=True)

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

    for fk_file in tqdm(files):
        load_and_plot(fk_file, params)



# Set your parameters here
params = {
    'smin': -50,
    'smax': 50,
    'cap_find': 144,
    'cap_fave': 6,
    'nsamp': 600,
    'dt': 1
}

# Call the function with the desired directory
process_directory('./fk_body_wave/', params)
