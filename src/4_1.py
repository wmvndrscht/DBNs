from util import *
from rbm import RestrictedBoltzmannMachine 
import math

# The aim is to use a mini-bach size = 20, epochs = 20, track the 
# reconstruction error


if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")
    
    batch_size = 10
    ndim_hidden = 500
    n_train = 60000
    n_iterations = 60000/batch_size #math.floor(60000/batch_size)
    epochs = 10;

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=ndim_hidden,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=batch_size
    )
    
    recon_err = []
    for i in range(epochs):
        rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)
        recon_err += rbm.get_err_rec()
	
    x_axis = np.arange(0,epochs,1.0/n_iterations)
    plt.title("Reconstruction Error along Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error")
    plt.plot(x_axis, recon_err)
    plt.savefig('4_1.png')
    plt.show()

