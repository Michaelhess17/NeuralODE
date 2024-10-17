def get_bootstrapped_sample(variable,nboot=10000):
    '''
    This function performs a hierarchical bootstrap on the data present in 'variable'.
    This function assumes that the data in 'variable' is in the format of a 2D array where
    the rows represent the higher level (in this case, each row is a separate neuron) and
    the number of columns represent repetitions within that level (trials).
    '''
    bootstats = np.zeros(nboot)
    for i in np.arange(nboot):
        temp = []
        num_lev1 = np.shape(variable)[0]
        num_lev2 = np.shape(variable[0])[0]
        rand_lev1 = np.random.choice(num_lev1,num_lev1)
        for j in rand_lev1:
            rand_lev2 = np.random.choice(num_lev2,num_lev2)
            temp.append(variable[j][rand_lev2])
        
        #Note that this is the step at which actual computation is performed. In all cases for these simulations
        #we are only interested in the mean. But as elaborated in the text, this method can be extended to 
        #several other metrics of interest. They would be computed here:
        bootstats[i] = np.mean(temp)
        
    return bootstats
import numpy as np # type: ignore

def get_direct_prob(sample1, sample2):
    '''
    get_direct_prob Returns the direct probability of items from sample2 being
    greater than or equal to those from sample1.
       Sample1 and Sample2 are two bootstrapped samples and this function
       directly computes the probability of items from sample 2 being greater
       than or equal to those from sample1. Since the bootstrapped samples are
       themselves posterior distributions, this is a way of computing a
       Bayesian probability. The joint matrix can also be returned to compute
       directly upon.
    '''
    joint_low_val = min([min(sample1),min(sample2)])
    joint_high_val = max([max(sample1),max(sample2)])
    
    p_joint_matrix = np.zeros((100,100))
    p_axis = np.linspace(joint_low_val,joint_high_val,num=100)
    edge_shift = (p_axis[2] - p_axis[1])/2
    p_axis_edges = p_axis - edge_shift
    p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

    #Calculate probabilities using histcounts for edges.

    p_sample1 = np.histogram(sample1,bins=p_axis_edges)[0]/np.size(sample1)
    p_sample2 = np.histogram(sample2,bins=p_axis_edges)[0]/np.size(sample2)

    #Now, calculate the joint probability matrix:
    # print(np.arange(np.shape(p_joint_matrix)[0]))
    for i in np.arange(np.shape(p_joint_matrix)[0]):
        for j in np.arange(np.shape(p_joint_matrix)[1]):
            p_joint_matrix[i,j] = p_sample1[i]*p_sample2[j]
            
    #Normalize the joint probability matrix:
    p_joint_matrix = p_joint_matrix/np.sum(p_joint_matrix)
    
    #Get the volume of the joint probability matrix in the upper triangle:
    p_test = np.sum(np.triu(p_joint_matrix))
    
    return p_test, p_joint_matrix

def get_bootstrapped_sample_with_error(variable, errors, nboot=10000):
    #     '''
    #     This function performs a hierarchical bootstrap on the data present in 'variable'.
    #     This function assumes that the data in 'variable' is in the format of a 2D array where
    #     the rows represent the higher level (in this case, each row is a separate neuron) and
    #     the number of columns represent repetitions within that level (trials).
    #     '''
    bootstats = np.zeros(nboot)
    for i in np.arange(nboot):
        temp = []
        num_lev1 = np.shape(variable)[0]
        rand_lev1 = np.random.choice(num_lev1,num_lev1)
        for j in rand_lev1:
            var = variable[j]
            error = errors[j]
            # rand_lev2 = np.random.normal(var,error, size=np.int64(np.sqrt(nboot)))
            rand_lev2 = np.random.uniform(var-error, var+error, size=np.int64(np.sqrt(nboot)))
            temp.append(rand_lev2)

        #Note that this is the step at which actual computation is performed. In all cases for these simulations
        #we are only interested in the mean. But as elaborated in the text, this method can be extended to 
        #several other metrics of interest. They would be computed here:
        bootstats[i] = np.mean(temp)
    return bootstats