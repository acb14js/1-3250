import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt

def mainloop(leaky=False, dinit=False, neighbours=False, show=False):
    train = np.genfromtxt('digits/train.csv', delimiter=",")
    trainlabels = np.genfromtxt('digits/trainlabels.csv', delimiter=",")
    
    # number of pixels and number of training data
    [n, m] = np.shape(train) # 784 * 5000, features * samples
    eta = 5E-2 # Learning rate of the wining unit
    theta = 5E-5 # Learning rate of other units. An few orders of magnitude smaller.
    winit = 1 # This scales the random numbers
    alpha = 0.999 # This is just a graphing term.
    cost = 0

    tmax = 40000 # The amount of timesteps/iterations of the algorithm.
    digits = 15 # the amount of class reconstructions. 

    # Initialising from a random sample in the data
    if dinit:
        index = np.random.choice(n, digits)
        winit = train[index]
        print(winit)

    # Weight matrix (rows = output neurons, cols = input neurons)
    W = winit * np.random.rand(digits, n) # Random weights for the network
    normW = np.sqrt(np.diag(W.dot(W.T)))
    # reshape normW into a numpy 2d array
    normW = normW.reshape(digits, -1)

    #W = W / np.matlib.repmat(normW.T,n,1).T    # normalise using repmat
    # normalise using numpy broadcasting -  http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
    W = W / normW # divide the weights by the magnitude.
    # 15 * 784, d*n

    counter = np.zeros((1, digits))              # counter for the number of times the neuron wins
    dead_units = None                            # Init the dead units list

    # running avg of the weight change over time
    wCount = np.ones((1, tmax+1)) * 0.25  # init the weight change over time

    if show:
        # NBVAL_SKIP
        plt.ion()                                   # interactive mode on
        plt.close('all')

        # counter for the rows of the subplot
        yl = int(round(digits/5)) 
        if digits % 5 != 0:
            yl += 1

        fig_neurs, axes_neurs = plt.subplots(
            yl, 5, figsize=(5, 4))  # fig for the output neurons
        fig_stats, axes_stats = plt.subplots(
            5, 1, figsize=(5, 7))   # fig for the learning stats
        
    for t in range(1, tmax): 
        # get a randomly generated index in the input range
        i = math.ceil(m * np.random.rand())-1 
        # pick a training instance using the random index
        x = train[:, i] # randomly choose a character from the dataset

        # This is the response from the units
        # Referred to as the pattern matrix
        # It is being normalised by the amount of digits.
        h = W.dot(x)/digits                     # get output firing
        h = h.reshape(h.shape[0], -1)            # reshape h into a numpy 2d array

        xi = np.random.rand(digits, 1) / 200 # THIS IS THE NOISE
        # get the max in the output firing vector + noise
        # This is the output of the winning unit
        output = np.max(h+xi) # Noise is added to pattern vector. 
        # get the index of the firing neuron
        k = np.argmax(h+xi) # This is the winning units index.

        # increment counter for winner neuron, 
        counter[0, k] += 1

        # calculate the change in weights for the k-th output neuron
        dw = eta * (x.T - W[k, :]) # Change in weights is based upon the input digit and the previous weights
        # get closer to the input (x - W)

        # % weight change over time (running avg)
        wCount[0, t] = wCount[0, t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # This is te exponential decay shown in the graph.

        # weights for k-th output are updated
        W[k, :] +=  dw # Finally add the weights to produce the weight change.
  
        # Identify the dead units
        mean = np.mean(counter[0])
        std = np.std(counter[0])
        deviation = (counter[0] - mean)/std
        dead_units = [x for x in range(len(W)) if deviation[x] < -1]

        if leaky: # Update the other weights
            theta_dw_lower = theta * (x.T - W[:k, :])
            theta_dw_higher = theta * (x.T - W[k+1:, :])
            W[:k, :] += theta_dw_lower
            W[k+1:, :] += theta_dw_higher
        
        if neighbours:
            # TODO implement the neighbours function
            print("neighbours")

        cost += 0.5*(W[k, :] - x)**2

        if not (t % 300 or t == 1) and show:
            for ii in range(yl):
                for jj in range(5):
                    if 5*ii+jj < digits:
                        prototypes = W[5*ii+jj, :]
                        output_neuron = prototypes.reshape((28, 28), order='F')
                        axes_neurs[ii, jj].clear()
                        axes_neurs[ii, jj].imshow(output_neuron,
                                                interpolation='nearest', cmap='inferno')

            # plot stats
            axes_stats[0].clear()
            axes_stats[0].bar(np.arange(1, digits+1), np.reshape(h, digits),
                            align='center', color='#57106e')
            axes_stats[0].set_xticks(np.arange(1, digits+1))
            axes_stats[0].relim()
            axes_stats[0].autoscale_view(True, True, True)

            axes_stats[1].clear()
            axes_stats[1].imshow(x.reshape((28, 28), order='F'),
                                interpolation='nearest', cmap='inferno')
            axes_stats[1].get_xaxis().set_ticks([])
            axes_stats[1].get_yaxis().set_ticks([])

            axes_stats[2].clear()
            axes_stats[2].imshow(W[k, :].reshape((28, 28), order='F'),
                                interpolation='nearest', cmap='inferno')
            axes_stats[2].get_xaxis().set_ticks([])
            axes_stats[2].get_yaxis().set_ticks([])

            axes_stats[3].clear()
            axes_stats[3].plot(wCount[0, 2:t+1], linewidth=2.0, label='rate')
            axes_stats[3].set_ylim([-0.001, 0.3])
            axes_stats[3].legend()

            axes_stats[4].clear()
            axes_stats[4].bar(np.arange(1, digits+1), np.reshape(counter.T, digits),
                            align='center', color='#57106e')

            print(dead_units)
            
            plt.show()
            plt.pause(0.0001)

    # Saving the figures ----------------------------------------
    plt.waitforbuttonpress()
    fig_neurs.savefig('fig_neurs.pdf')
    fig_stats.savefig('fig_stats.pdf')

    axes_stats[3].get_window_extent().transformed(fig_stats.dpi_scale_trans.inverted())
    fig_stats.savefig('average_weight_changes.pdf')
    # might need to include bbox_inches=extent.expanded(1.1, 1.2)

    plt.clf()
    plt.cla()
    plt.close()

    # Correlation matrix of the prototypes
    corr = np.corrcoef(W)
    plt.figure()
    plt.imshow(corr, interpolation="nearest", cmap="inferno")
    plt.title('Correlation Of Prototypes')
    plt.ylabel("prototype")
    plt.xlabel("prototype")
    plt.colorbar()
    plt.savefig('correlation_matrix.pdf')

    print(str(len(counter[0] - len(dead_units) + " prototypes were found.")))

def main():
    mainloop(leaky=True, dinit=False)

if __name__ == '__main__':
    main()
