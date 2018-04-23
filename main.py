import numpy as np
# Local packages
import pca_step
import competitive_learning

def main():
    # The PCA algorithm ----------------------------------
    pca_step.main()

    # The Standard Competitive Algorithm----------------------------------
    plain = np.zeros(2)
    sample_init = np.zeros(2)
    leaky = np.zeros(2)
    both = np.zeros(2)

    for x in range(10):
        plain += competitive_learning.mainloop()/10 # Plain
        sample_init += competitive_learning.mainloop(dinit=True)/10  # Init from samples
        leaky += competitive_learning.mainloop(leaky=True)/10 # Leaky
        both += competitive_learning.mainloop(leaky=True, dinit=True)/10 # Both

    print("Name . Avg Dead Units . Cost")
    print("plain", end="")
    print(plain)
    print("sample_init", end="")
    print(sample_init)
    print("leaky", end="")
    print(leaky)
    print("both", end="")
    print(both)

if __name__ == '__main__':
    main()
