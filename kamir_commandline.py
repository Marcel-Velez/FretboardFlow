import argparse
import os
import numpy as np
from scipy.signal import stft, istft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import copy
from tqdm import tqdm

def setup_interference_matrix(guitar_init, selection_threshold, I, J):
    # Frequency-dependent interference matrix

    if guitar_init:
        sources_names = ['String1', 'String2', 'String3', 'String4', 'String5', 'String6']
        # Initialize interference matrix if not good_init
        # A simple model: strong diagonal for main string signals, smaller values for adjacent strings
        L0 = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        L = np.maximum(L0, selection_threshold)
        J = L.shape[1]
    else:
        sources_names = None
        L = np.zeros((I, J))
        step = I / J
        pos = np.round((np.arange(0, J) * step)).astype(int)
        pos = np.vstack((pos, np.full(J, I)))

        for j in range(J):
            start_idx = max(1, pos[0, j] - 1)
            if j + 1 < J:
                end_idx = min(I, pos[0, j + 1])
            L[start_idx:end_idx, j] = 1

        L = np.maximum(selection_threshold, L)
        L0 = copy.deepcopy(L)
        J = L.shape[1]

    return L, L0, J, sources_names


def retrieve_files(datadir):
    # Listing files # might need to adjust this so it can iterate over all files per song with 6 string recordings
    filenames = sorted(
        [f for f in os.listdir(datadir) if (f.endswith('.wav') and f.startswith(('1_', '2_', '3_', '4_', '5_', '6_')))])
    if not filenames:
        print("No file found. Check your directory. Aborting.")
    else:
        pass
        # print(filenames)
    I = len(filenames)

    return filenames, I


def load_files(filenames, datadir, Lmax=500):
    signals = []
    for filename in filenames:
        fs, data = wavfile.read(os.path.join(datadir, filename))
        if data.ndim > 1:
            data = data[:, 0]  # Use the first channel
        truncated = data[:min(len(data), int(Lmax * fs))]
        signals.append((fs, truncated))

    return signals


def stft_files(signals, nfft, overlap):
    # STFT
    X = []
    for fs, signal in signals:
        f, t, Zxx = stft(signal, fs, nperseg=nfft, noverlap=int(nfft * overlap))  # originally generated line
        X.append(Zxx)
    X = np.stack(X, axis=-1)  # Shape: (freq, time, mic)

    F, T, I = X.shape

    return X, F, T, I


def step_3_automatic_handling_of_equalization(X, F, I, alpha, fs, plot=False):
    gains = np.zeros((F, I))
    for i in range(I):  # For each mic
        V = np.abs(X[:, :, i]) ** alpha
        gains[:, i] = np.quantile(V, 0.05, axis=1) ** (1 / alpha)

    # Perform element-wise division
    gains_expanded = 1 / gains[:, np.newaxis, :]
    X = X * gains_expanded
    if plot:
        plt.loglog(np.linspace(0, fs / 2, X.shape[0]), gains)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Noise floor')
        plt.grid()
        plt.title('Noise floor for all channels')
        plt.show()
    return X, gains


def step_4_initialize_kernel_backfitting(X, F, T, J, alpha, good_init, L, proximityKernel, selection_threshold):
    if not isinstance(proximityKernel, int):
        midposKernel = round(len(np.where(proximityKernel)[0]) / 2)
    else:
        midposKernel = 1

    # Initialize the sources as the observation
    P = np.zeros((F, T, J))
    V = np.abs(X) ** alpha
    for j in range(J):
        if not good_init:
            channel_mask = np.median(L[:, :, j], axis=0) > selection_threshold
            P[:, :, j] = np.mean(V[:, :, channel_mask], axis=2)
        else:
            P[:, :, j] = np.mean(V[:, :, np.where(L0[:, j])[0]], axis=2)  # was np.where(L0[:,j])...

        if not isinstance(proximityKernel, int) and proximityKernel.size > 1:
            P[:, :, j] = rank_filter(P[:, :, j], rank=midposKernel, footprint=proximityKernel)

    return P, V, midposKernel


def step_5_kernel_backfitting(X, niter, J, L, selection_threshold, L0, F, T, slope, thresh, fs, nfft, overlap, suffix,
                              source_names, alpha, proximityKernel, midposKernel, learn_L, niter_L, beta, V, P, good_init, approx, outdir, gains, normalize=True):
    for it in tqdm(range(niter + 1)):
        # niter+1 for rendering and wavwriting

        for j in range(J):

            MaxInterference = np.max(np.median(L[:, :, j], axis=0))
            if not good_init:  # or it<= niter:
                closemics = np.where(np.median(L[:, :, j], axis=0) > (selection_threshold * MaxInterference))[0]
            else:
                closemics = np.where(L0[:, j])[0]

            # get the image of this source in these channels
            Y = np.zeros((F, T, len(closemics)), dtype=complex)
            for n, mic in enumerate(closemics):
                # compute model for this channel
                model = np.zeros((F, T)) + np.finfo(float).eps
                for j2 in range(J):
                    model += L[:, mic, j2][:, np.newaxis] * P[:, :, j2]

                # compute wiener filters to separate image of j in this chann
                W = (L[:, mic, j].reshape(-1, 1) * P[:, :, j]) / model

                # if we do the approx, do the logit stuff
                if approx and it == 0:
                    W = 1 - 1 / (1 + np.exp(slope * (W - thresh)))

                # apply the Wiener gain
                Y[:, :, n] = W * X[:, :, mic]

            # Y is the image of the source in its channels of importance
            # if we are finished, just render Y and wavwrite it
            if it == niter:  # Final iteration: Write separated outputs

                # for each channel, make a istft
                for n in range(len(closemics)):
                    # Y(:,:, n) = bsxfun( @ times, Y(:,:, n), gains(:, closemics(n))); in matlab converts to:
                    # Y[:, :, n] = Y[:, :, n] * gains[:, closemics[n]][:, np.newaxis]
                    t, waveform = istft(Y[:, :, n], fs, nperseg=nfft, noverlap=int(nfft * overlap))

                    print(f"# saving file {suffix}_source_{j}.wav ###")

                    # added normalization to prevent clipping, for some reason the original code caused the louder strings to clip after normalizing it before the start of the algorithm
                    if normalize:
                        waveform = waveform / np.max(np.abs(waveform))
                        waveform = (waveform * 32767).astype(np.int16)
                        wavfile.write(
                            os.path.join(outdir, f"{suffix}_source_{j}.wav"),
                            fs,
                            waveform.astype(np.int16),
                        )
                    else:
                        wavfile.write(
                            os.path.join(outdir, f"{suffix}_source_{j}.wav"),
                            fs,
                            waveform.astype(np.int16),
                        )

                    #
                    # wavfile.write(
                    #     os.path.join(outdir, f"{suffix}_source_{j}_new.wav"),
                    #     fs,
                    #     waveform.astype(np.int16),
                    # )
            # Compute average spectrogram on current relevant channels
            first_step = np.abs(Y) ** alpha
            permuted_L = 1 / L[:, closemics, j][:, np.newaxis, :]

            P[:, :, j] = np.mean(first_step * permuted_L, axis=2)

            # Apply median filter if needed
            if np.size(proximityKernel) > 1:
                P[:, :, j] = rank_filter(P[:, :, j], rank=midposKernel, footprint=proximityKernel)

        if learn_L and (it <= niter):
            # now learn gains
            for it_l in range(niter_L):
                if niter_L > 0:
                    print(f' updating gains: [{it_l}/{niter_L}]', end='')
                else:
                    print('# updating gains #', end='')

                for i in range(I2):
                    model = np.ones((F, T)) * np.finfo(float).eps
                    for j2 in range(J):
                        model += L[:, i, j2].reshape(-1, 1) * P[:, :, j2]

                    for j in range(J):
                        if not np.isnan(beta):
                            # classical beta-divergence
                            num = np.finfo(float).eps + np.sum(model ** (beta - 2) * V[:, :, i] * P[:, :, j], axis=1)
                            denum = np.finfo(float).eps + np.sum(
                                model ** (beta - 1) * np.maximum(P[:, :, j], np.finfo(float).eps), axis=1)
                        else:
                            # with L_alpha distortion
                            distortion = np.abs(np.finfo(float).eps + model - V[:, :, i]) ** (alpha - 2)
                            num = np.finfo(float).eps + np.sum(V[:, :, i] * distortion * P[:, :, j], axis=1)
                            denum = np.finfo(float).eps + np.sum(model * distortion * P[:, :, j], axis=1)
                        L[:, i, j] = L[:, i, j] * num / denum

                LsumF = np.sum(L, axis=2)
                L = L / LsumF[:, :, np.newaxis]
                L = np.maximum(L, minleakage)

            plot_average_interference_matrix(L)




def plot_average_interference_matrix(L):
    """
    Plots the average interference matrix.

    Args:
    L: Interference matrix.
    """

    plt.figure(10)
    plt.clf()
    plt.imshow(np.mean(L, axis=0))
    plt.xlabel('sources j')
    plt.ylabel('channels i')
    plt.colorbar()
    plt.title('Average Interference matrix $\lambda_{ij}$', fontsize=16)
    plt.show()  # Use plt.show() instead of drawnow()
    print('    done')

def main():
    parser = argparse.ArgumentParser(description='KAMIR command line arguments')
    parser.add_argument('--datadir', type=str, required=True, help='Input directory containing wav files')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--suffix', type=str, required=True, help='Suffix for output files')


    parser.add_argument('--I', type=int, default=6, help='Number of microphones')
    parser.add_argument('--J', type=int, default=6, help='Number of voices')

    parser.add_argument('--nfft', type=int, default=8192, help='Length of FFT window')
    parser.add_argument('--overlap', type=float, default=0.9, help='Overlap of adjacent windows')
    parser.add_argument('--Lmax', type=int, default=500, help='Maximum length in seconds')

    parser.add_argument('--minleakage', type=float, default=0.05, help='Minimum amount of leakage')

    parser.add_argument('--selection_threshold', type=float, default=0.8, help='Threshold to select channel for fitting')
    parser.add_argument('--alpha', type=int, default=2, help='Alpha parameter (2=gaussian)')

    parser.add_argument('--approx', type=bool, default=True, help='Use logistic approximation')

    parser.add_argument('--good_init', type=bool, default=False, help='User defined initialization')
    parser.add_argument('--guitar_init', type=bool, default=True, help='Guitar initialization')

    parser.add_argument('--niter_L', type=int, default=3, help='Number of iterations for learning L')

    parser.add_argument('--learn_L', type=bool, default=False, help='Learn L')
    parser.add_argument('--beta', type=float, default=0, help='Beta divergence for fitting L')

    parser.add_argument('--proximityKernel', type=int, default=1, help='Kernel for median filtering')
    parser.add_argument('--plot_noise_floor', type=bool, default=False, help='Learn L')

    parser.add_argument('--normalize', type=bool, default=False, help='Learn L')

    args = parser.parse_args()

    # print(args.normalize)
    # print(args.good_init)
    #
    # exit()
    if args.approx:
        slope = 20
        thresh = 0.6
    else:
        niter = 5  # number of iterations

    if not args.good_init:
        J = args.J  # could also be 7 if we include overall recording # in source code set to 11 (for the 11 mics)
        niter_L = args.niter_L
    else:
        J = args.J
        niter_L = 1

    for subdir in tqdm(sorted(os.listdir(args.datadir))):  # Sorting for consistency
        subdir_path = os.path.join(args.datadir, subdir)

        # Ensure it's a directory
        if not os.path.isdir(subdir_path):
            continue

        # Create a corresponding output directory
        out_subdir = os.path.join(args.outdir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        print (" ### step 1: ", subdir, " #", end='')
        # step 1
        L, L0, J, sources_names = setup_interference_matrix( args.guitar_init, args.selection_threshold, args.I, J)

        print("# step 2 #", end="")
        # step 2 # retrieve files and load them and stft them
        filenames, I = retrieve_files(subdir_path)
        signals = load_files(filenames, subdir_path, args.Lmax)

        fs = signals[0][0]

        X, F, T, I = stft_files(signals, args.nfft, args.overlap)

        # put this somewhere between step 2 and 3
        L = np.tile(L[np.newaxis, :, :], (F, 1, 1))

        print("# step 3 #", end='')
        # step 3
        # if args.normalize:
        # X, gains = step_3_automatic_handling_of_equalization(X, F, I, args.alpha, fs, plot=args.plot_noise_floor)

        print("# step 4 #", end='')
        # step 4
        P, V, midposKernel = step_4_initialize_kernel_backfitting(X, F, T, J, args.alpha, args.good_init, L, args.proximityKernel, args.selection_threshold)

        # put this somewhere between step 4 and 5
        if args.approx:
            if args.learn_L:
                niter = 1
            else:
                niter = 0
        else:
            niter = None

        print("# step 5 #", end='')
        # step 5

        step_5_kernel_backfitting(X, niter, J, L, args.selection_threshold, L0, F, T, slope, thresh, fs, args.nfft,
                                  args.overlap, args.suffix, sources_names, args.alpha, args.proximityKernel,
                                  midposKernel, args.learn_L, niter_L, args.beta, V, P, args.good_init, args.approx,
                                  out_subdir, None, args.normalize)



if __name__ == '__main__':
    main()




