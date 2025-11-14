# WH-ESPC

The WH-ESPC model is a novel artificial intelligence model based on an enhanced Mixture-of-Experts (MOE) architecture, designed for the efficient and accurate three-class classification of healthy individuals, Preserved Ratio Impaired Spirometry (PRISm), and Chronic Obstructive Pulmonary Disease (COPD) using computed tomography (CT) imaging, thereby enabling the goal of early screening and treatment for COPD.

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability
# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 64
RNN_FC_dim = 32
# training parameters
k = 3             # number of target category
epochs = 80      # training epochs
batch_size = 34
learning_rate = 1e-4
log_interval = 2   # interval for displaying training info
#--------MOE--------
n_embed = 512  
embed_dim = 512
num_heads = 4
num_layers = 2
num_experts = 16  #
top_k = 4 
capacity_factor = 1 
alpha = 0.05  

We have placed the dataset in the DataCOPD folder; additionally, you will need to create a Label file.
