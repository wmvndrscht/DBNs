from util import *

class RestrictedBoltzmannMachine():
    #For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=20):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        
        # initialise the weights to small random values N(0,0.01) 
        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.1
        
        self.momentum = 0.7

        self.print_period = 500
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 500, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        self.rec_err = []

        return

        
    def cd1(self,visible_trainset, n_iterations=3000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations):

            # Positive phase
            # initial values for a batch
            minibatch_ndx = int(it % (n_samples/self.batch_size))
            minibatch_end = min([(minibatch_ndx+1)*self.batch_size, n_samples])
            minibatch = visible_trainset[minibatch_ndx*self.batch_size:minibatch_end, :] 


            v_0 =  minibatch#visible_trainset[it*self.batch_size:(it+1)*self.batch_size][:] 
            # find h_0 
            prob_on_hidden_0, h_0 = self.get_h_given_v(v_0)

            # Negative phase
            # find v_k
            prob_on_visible_k, v_k = self.get_v_given_h(h_0)
            # find h_k
            prob_on_hidden_k, h_k = self.get_h_given_v(v_k)

            # updating parameters
            self.update_params(v_0=v_0,h_0=h_0,v_k=v_k,h_k=h_k) 
 
            # visualize once in a while when visible layer is input images
            
            if it % self.rf["period"] == 0 and self.is_bottom:
                
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # print progress
            
            if it % self.print_period == 0 :
                print("iteration=%7drecon_loss=%4.4f"%(it,np.linalg.norm((1.0/self.batch_size)*np.sum(v_0-v_k,axis=0))))

            self.rec_err.append(np.linalg.norm((1.0/self.batch_size)*np.sum(v_0-v_k,axis=0)))
        return
    

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """
        
        self.delta_bias_v = (1.0/self.batch_size)*self.learning_rate*np.sum((v_0-v_k).transpose(),axis=1)
        self.delta_bias_h = (1.0/self.batch_size)*self.learning_rate*np.sum((h_0-h_k).transpose(),axis=1)
        self.delta_weight_vh = (1.0/self.batch_size)*self.learning_rate*(np.dot(v_0.transpose(),h_0)-np.dot(v_k.transpose(),h_k))

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]
        
        # Update hidden states, p(h_j = 1) = logistic(b_j + sum_i(v_i*w_ij) 
        prob_on = sigmoid(np.tile(self.bias_h,(n_samples,1)) + np.dot(visible_minibatch,self.weight_vh))
        h = sample_binary(prob_on)
        return prob_on, h 


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            # [Labels | Data] compute with weights both normally
            total_in = np.tile(self.bias_v,(n_samples,1))+np.dot(hidden_minibatch,self.weight_vh.transpose())
            
            # split into sections for activations and sampling
            data_in = total_in[:,:-self.n_labels]
            label_in = total_in[:,-self.n_labels:]

            # data we use sigmoid and binary sampling
            data_prob_on = sigmoid(data_in)
            data_sampled = sample_binary(data_prob_on)

            # labels we use softmax and categorical sampling
            label_prob_on = softmax(label_in)
            label_sampled = sample_categorical(label_in)

            # concatenate results
            prob_on = np.concatenate((data_prob_on, label_prob_on), axis=1)
            v = np.concatenate((data_sampled, label_sampled), axis=1)
            
        else:

            prob_on = sigmoid(np.tile(self.bias_v,(n_samples,1))+np.dot(hidden_minibatch,self.weight_vh.transpose()))
            v = sample_binary(prob_on)
        
        return prob_on, v

    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """
    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]
    
        prob_on = sigmoid(np.tile(self.bias_h,(n_samples,1))+np.dot(visible_minibatch,self.weight_v_to_h))
        h = sample_binary(prob_on)

        return prob_on, h


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        prob_on = sigmoid(np.tile(self.bias_v,(n_samples,1))+np.dot(hidden_minibatch,self.weight_h_to_v))
        v = sample_binary(prob_on)

        return prob_on, v
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        self.delta_weight_h_to_v = (1.0/self.batch_size)*self.learning_rate*np.dot((trgs-preds).transpose(),inps).transpose()
        self.delta_bias_v = (1.0/self.batch_size)*self.learning_rate*np.sum(trgs-preds,axis=0)
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        self.delta_weight_v_to_h = (1.0/self.batch_size)*self.learning_rate*np.dot((trgs-preds).transpose(),inps).transpose()
        self.delta_bias_h = (1.0/self.batch_size)*self.learning_rate*np.sum(trgs-preds,axis=0)

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
    def get_err_rec(self):
        """ Gets the reconstruction error and clear it"""
        aux = self.rec_err
        self.rec_err = []
        return aux
