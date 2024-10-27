import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default=2,\
                        help="Seed for the code")
    parser.add_argument('--ds', type= str, default="Delicious", \
                    help="dataset name")
# current options of dataset: 'Delicious' 'eron' 'Col5k' 'bibTex' 'eurLex'
    parser.add_argument('--fname', type= str, default="bs", \
                        help="Identifier For Script")
    parser.add_argument('--lambda_ker', type= float, default=0.1, \
                        help="Kernel Regularization")

    parser.add_argument('--gpu_id', type=int, default=0, \
                        help = "GPU ID")
    parser.add_argument('--debug_id', type=int, default=2, \
                        help = "Debug ID")
    parser.add_argument('--hidden_size', type=int, default=1024, \
                        help = "The hidden layer size")
    parser.add_argument('--seed0', type = int, default=1,\
                        help="Seed for loading")

# AL parameters    
    parser.add_argument('--train', type= float, default=0.01, \
                    help="train size")
    parser.add_argument('--pool', type= float, default=0.6, \
                    help="pool size")
    parser.add_argument('--test', type= float, default=0.2, \
                    help="test size") 

    parser.add_argument('--AL_rounds', type = int, default=10,\
                    help="Number of AL rounds")
    parser.add_argument('--al_mtd', type= str, default="bvs", \
                    help="Type of AL method")
    parser.add_argument('--AL_batch', type = int, default=100,\
                    help="Batch size")    
    
    parser.add_argument('--n_components', type = int, default=6,\
                    help="Number of mixture components")

    parser.add_argument('--wnew', type= float, default=10.0, \
                    help="Components update weight") 

# training parameters
    parser.add_argument('--optimizer', type= str, default="adam", \
                    help="Type of optimizer")
    parser.add_argument('--lr', type= float, default=1e-4, \
                    help="Learning rate")  
    parser.add_argument('--wd', type= float, default=0, \
                    help="Weight decay")  

# BM_sepNN model settings
    parser.add_argument('--m_hidden', type= int, default=1024, \
                    help="Number of nodes per hidden layer")
    parser.add_argument('--m_embed', type= int, default=1024, \
                help="Number of features in embedding space")
    parser.add_argument('--m_activation', type= str, default="softplus", \
                    help="Type of activation function used in model")
    parser.add_argument('--m_drop_p', type= float, default=0.1, \
                    help="Dropout ratio")  
    parser.add_argument('--m_decoder_type', type= str, default="deep", \
                    help="Type of decoder for components")
    parser.add_argument('--m_cons', type= bool, default=False, \
                    help="Whether constrain the output")

    

# experiment settings
    parser.add_argument('--pretrain_loss', type= str, default="NIG", \
                    help="Type of loss used in weights pretraining step")
    parser.add_argument('--pretrain_epochs', type = int, default=5000,\
                    help="Number of weights pretraining epochs")

    parser.add_argument('--tr_rounds', type = int, default=100,\
                    help="Number of alternating training rounds")
    parser.add_argument('--bookkeep', type = str, default='No',\
                    help="Type of book keeping of components")

    parser.add_argument('--bk_w', type = float, default=1,\
                    help="Book keeping update weight")

    parser.add_argument('--msm_step', type= bool, default=False, \
                    help="Whether use msm to train weights")
    parser.add_argument('--msm_epochs', type = int, default=1,\
                    help="Number of MSM training epochs (per step)")

    parser.add_argument('--opt_mu', type= bool, default=False, \
                    help="Whether optimize new alpha")

# label step training settings
    parser.add_argument('--l_loss', type= str, default="NON", \
                    help="Type of loss used in joint label training step")
    parser.add_argument('--l_epochs', type = int, default=1,\
                    help="Number of joint label training epochs (per step)")

# weights only step training settings
    parser.add_argument('--pi_loss', type= str, default="NIG", \
                    help="Type of loss used in weights-only training step")
    parser.add_argument('--pi_epochs', type = int, default=1,\
                    help="Number of weights-only training epochs (per step)")

# sampling parameters
    parser.add_argument('--s_lambda', type = float, default=1,\
                    help="Coefficient for covariance 1")
    parser.add_argument('--s_eta', type = float, default=0.1,\
                    help="Coefficient for theta dif ")
    args = parser.parse_args()
    return args, args.seed