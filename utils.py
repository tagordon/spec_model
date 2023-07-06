import corner
import numpy as np

def plot_nf_convergence(axs, model):
    
    if model.trained_sampler is None:
        raise Exception('sampler must first be trained by calling Model.train(data, initial_parameters)')
     
    out = model.trained_sampler.get_sampler_state(training=True)
    global_accs = np.array(out['global_accs'])
    local_accs = np.array(out['local_accs'])
    loss_vals = np.array(out['loss_vals'])
    
    axs[0].set_title("NF loss")
    axs[0].plot(loss_vals.reshape(-1))
    axs[0].set_xlabel("iteration")
    
    axs[1].set_title("Local Acceptance")
    axs[1].plot(local_accs.mean(0))
    axs[1].set_xlabel("iteration")
    
    axs[2].set_title("Global Acceptance")
    axs[2].plot(global_accs.mean(0))
    axs[2].set_xlabel("iteration")
    
def plot_nf_samples(model):
    
    if model.trained_sampler is None:
        raise Exception('sampler must first be trained by calling Model.train(data, initial_parameters)')
        
    out = model.trained_sampler.get_sampler_state(training=True)
    chains = np.array(out['chains'])
        
    nf_samples = np.array(model.trained_sampler.sample_flow(1000)[1])
    
    corner.corner(
        chains.reshape(-1, len(model.labels)), 
        labels=model.labels,
    );