from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_rectg = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        n_labels = true_lbl.shape[1]

        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
       
        # Bottom-up
        # vis->hid (sampled data)
        hidden_up = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[1]
        # hid->pen (sampled data)
        pen_up = self.rbm_stack["hid--pen"].get_h_given_v_dir(hidden_up)[1]
        # concatenate with labels
        pen_data_labels = np.concatenate((pen_up, lbl), axis=1)

        # perform Gibb sampling n_gibbs_recog times
        for _ in range(self.n_gibbs_recog):
            # get top hidden sampled data
            top = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_data_labels)[1]
            pen_data_labels = self.rbm_stack["pen+lbl--top"].get_v_given_h(top)[1]

        predicted_lbl = pen_data_labels[:, -n_labels:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        n_labels = true_lbl.shape[1]

        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl
        pen_data = sample_binary(0.5*np.ones((n_sample, self.sizes['pen'])))
        pen = np.concatenate((pen_data, lbl), axis=1)

        # perform gibbs sampling at top, propogate result down each time for
        # visualisation
        for _ in range(self.n_gibbs_gener):
            # gibbs sample at top
            top = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen)[1]
            pen = self.rbm_stack["pen+lbl--top"].get_v_given_h(top)[1]

            # genertate to visible
            pen_data = pen[:, :-n_labels]
            # pen -> hid
            hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_data)[1]
            # hid -> vis
            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid)[1] # could choose probs?
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """            
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()
            """ 
            CD-1 training for hid--pen 
            """
            hid_trainset = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[1]
            self.rbm_stack["hid--pen"].cd1(hid_trainset,n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")            

            print ("training pen+lbl--top")
            self.rbm_stack["hid--pen"].untwine_weights()
            """ 
            CD-1 training for pen+lbl--top 
            """
            pen_trainset_data = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_trainset)[1]
            pen_trainset = np.concatenate((pen_trainset_data,lbl_trainset),axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(pen_trainset,n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]
            n_samples = self.n_samples
            n_labels = lbl_trainset.shape[1]
            
            for it in range(n_iterations):            
                                
                """ 
                wake-phase : drive the network bottom-to-top using visible and label data
                """
                hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[1]
                pen_data = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid)[1]
                pen_data_labels = np.concatenate((pen_data,lbl_trainset),axis=1)
                top = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_data_labels)[1]

                v0_top = pen_data_labels
                h0_top = top

                """
                alternating Gibbs sampling in the top RBM : also store neccessary information for learning this RBM
                """
                for it in range(self.n_gibbs_wakesleep):
                    pen_data_labels = self.rbm_stack["pen+lbl--top"].get_v_given_h(top)[1]
                    top = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_data_labels)[1]
                
                vk_top = pen_data_labels
                hk_top = top

                """
                sleep phase : from the activities in the top RBM, drive the network top-to-bottom
                """
                pen_down = pen_data_labels[:, :-n_labels]
                hid_down = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_down)[1]
                vis_down = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid_down)[1]

                """
                predictions : compute generative predictions from wake-phase activations, 
                              and recognize predictions from sleep-phase activations
                """
                vis_pred_gen = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid)[0]
                hid_pred_gen = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen)[0]

                pen_pred_rec = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_down)[0]
                hid_pred_rec = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_down)[0]

                """ 
                update generative parameters :
                here you will only use "update_generate_params" method from rbm class
                """
                self.rbm_stack["vis--hid"].update_generate_params(hid,vis,vis_pred_gen)
                self.rbm_stack["hid--pen"].update_generate_params(pen,hid,hid_pred_gen)

                """ 
                update parameters of top rbm:
                here you will only use "update_params" method from rbm class
                """
                self.rbm_stack["pen+lbl--top"].update_params(v0_top,h0_top,vk_top,hk_top)
                
                """ 
                update recognize parameters :
                here you will only use "update_recognize_params" method from rbm class
                """
                self.rbm_stack["hid--pen"].update_recognize_params(hid_down,pen_down,pen_pred_rec)
                self.rbm_stack["vis--hid"].update_recognize_params(vis_down,hid_down,hid_pred_rec)

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
