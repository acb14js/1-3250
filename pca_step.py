import numpy as np
import math
import numpy.matlib
import matplotlib.patches as mpatches # Graph legends
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # load the dataset
    train = np.genfromtxt('digits/train.csv', delimiter=",")
    trainlabels = np.genfromtxt('digits/trainlabels.csv', delimiter=",")

    # pixels * observations
    [n, m] = np.shape(train)  

    # The letter type is dependent on the pixels.
    pixels = 28

    # This is calculating the magnitude
    normT = np.sqrt(np.diag(train.T.dot(train)))

    # Normalising the data
    train = train / np.matlib.repmat(normT.T, n, 1)

    # Putting the data in terms of features.
    data = train.T

    # number of Principal Components to save
    nPC = 6

    # n is the number of features.
    PCV = np.zeros((n, nPC))

    # Center data around the mean
    meanData = np.matlib.repmat(data.mean(axis=0), m, 1)
    data = data - meanData # This simply equates to normalising by the mean

    # Compute the covariance matrix in terms of observations and features.
    C = np.cov(data.T)

    # Solve an ordinary or generalized eigenvalue problem
    eigen_val, eigen_vec = np.linalg.eigh(C)

    # sorting the eigenvalues in descending order
    idx = np.argsort(eigen_val)
    idx = idx[::-1] # Reverses the array
    
    # Use the indices to sort the eigenvectors 
    eigen_vec = eigen_vec[:, idx] 
    
    # SORT BY EIGENVALUES and keep EIGENVECTORS
    eigen_val = eigen_val[idx] # new ordering

    # save only the most significant eigen vectors
    PCV[:, :nPC] = eigen_vec[:, :nPC]

    # apply transformation on the data
    FinalData = data.dot(PCV)

    # find indexes of data for each digit
    zeroData = (trainlabels == 0).nonzero()
    twoData = (trainlabels == 2).nonzero()
    fourData = (trainlabels == 4).nonzero()
    sevenData = (trainlabels == 7).nonzero()
    eightData = (trainlabels == 8).nonzero()

    # Init labels for the figures
    def_labels = 'Zeros Twos Fours Sevens Eights'.split()
    color = 'r. g. m. y. b.'.split()
    allData=[zeroData,twoData,fourData,sevenData,eightData]

    def xyzplot(x, y, z, ax=None, save=False):

        if ax == None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.gca(projection='3d')

        # Generalise the plotting of datapoints
        i=0 
        for dataName in allData:
            xcomp = FinalData[dataName, x].flatten()
            ycomp = FinalData[dataName, y].flatten()
            zcomp = FinalData[dataName, z].flatten()
            ax.plot(xcomp, ycomp, zcomp, color[i], label=def_labels[i])
            i+=1

        # Add the labels to the figures
        label = str(x)+', '+str(y)+' and '+str(z)+' PC'
        ax.set_title(label)
        ax.set_xlabel(str(x)+' Principle Component')
        ax.set_ylabel(str(y)+' Principle Component')
        ax.set_zlabel(str(z)+' Principle Component')

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels)

        # plt.show()
        if save:
            fname = str(x)+str(y)+str(z)+'.pdf'
            plt.savefig(fname, bbox_inches='tight')

    # xyzplot(1,2,3)
    # xyzplot(2,3,4)
    # xyzplot(1,3,4)
    # xyzplot(2,3,5)
    # xyzplot(3,4,5)

    initial_plots = [(1,2,3),(2,3,4),(1,3,4),(2,3,5),(3,4,5)]

    def iter_plot(d_list, indie=False):
        # if not indie:
        #     columns = math.ceil(len(d_list)/3)
        #     sub_plt_shape = str(columns) + '3'
        #     fig, axarr = plt.subplots(columns, 3, subplot_kw = dict(projection='polar'))
        #     axarr = axarr.flatten().tolist()

        #     count=1
        #     for xyz in d_list:
        #         x, y, z = xyz
                
        #         xyzplot(x,y,z, ax=axarr[count-1])
        #         count+=1
        #     plt.show()
        # else:
        for xyz in d_list:
            print("Plotting")
            x,y,z = xyz
            xyzplot(x,y,z)
            # plt.show()
            
    iter_plot(initial_plots, indie=False)

if __name__ == '__main__':
    main()
