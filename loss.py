import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from edl_pytorch import NormalInvGamma, evidential_regression

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


# define evidence operations
def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    #print('first term',first_term.shape,first_term)
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    #print('digamma(alpha)',torch.digamma(alpha).shape,torch.digamma(alpha))
    #print('second term',second_term.shape,second_term)
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

#special
    y[y>1e-20] =1
    y[y<=1e-20] = 0
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * 10*kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    y[y>0.1] =1
    y[y<=0.1] = 0
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def nig_loss(y, outputs, epoch_num, num_classes, annealing_step, device=None,lamb=1e-2):
    if not device:
        device = get_device()
    y = y.to(device)
    # alpha = alpha.to(device)
    loss = evidential_regression( outputs, y,lamb)

#     annealing_coef = torch.min(
#         torch.tensor(1.0, dtype=torch.float32),
#         torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
#     )

# #special
#     y[y>1e-20] =1
#     y[y<=1e-20] = 0
#     kl_alpha = (alpha - 1) * (1 - y) + 1
#     kl_div = annealing_coef * 10*kl_divergence(kl_alpha, num_classes, device=device)
    return loss

def non_mse_loss(y, outputs, epoch_num, num_classes, annealing_step, device=None,lamb=1e-2):
    if not device:
        device = get_device()
    y = y.to(device)
    # alpha = alpha.to(device)
    # non_mse = nn.MSELoss()
    non_mse = nn.MultiLabelSoftMarginLoss()

    # print("outputs",outputs[0].shape,outputs)
    # print("y",y)
    loss = non_mse( outputs[0], y)

    return loss

def ml_nn_loss(y, outputs, epoch_num, num_classes, \
                    annealing_step, model,device=None,\
                        lamb=1e-2,num_l=6, wnew=100):
    if not device:
        device = get_device()
    y = y.to(device)
    # alpha = alpha.to(device)
    # non_mse = nn.MSELoss()
    non_mse = nn.MultiLabelSoftMarginLoss()

    # print("outputs",outputs[0].shape,outputs)
    # print("y",y)
    pi = outputs
    loss = non_mse( pi.view(-1,num_l), y)
    return loss

def nonsep_mse_loss(y, outputs, epoch_num, num_classes, \
                    annealing_step, model,device=None,\
                        lamb=1e-2,num_l=6, wnew=100):
    if not device:
        device = get_device()
    y = y.to(device)
    # alpha = alpha.to(device)
    # non_mse = nn.MSELoss()
    non_mse = nn.MultiLabelSoftMarginLoss()

    # print("outputs",outputs[0].shape,outputs)
    # print("y",y)
    pi = outputs[0]
    theta = outputs[1]
    # print('theta',theta.shape,theta)
    a_p = theta[:,:num_classes*num_l]
    b_p = theta[:,num_classes*num_l:]
    # print('comp',model.comp_0.shape)

    a_sl = model.comp_0[:,:num_classes*num_l]
    b_sl = model.comp_0[:,num_classes*num_l:]
    # print('a_sl',a_sl.shape)

    a_new = a_p/wnew+a_sl
    b_new = b_p/wnew+b_sl
    # a_new = a_sl
    # b_new = b_sl
    theta_new = a_new/(a_new+b_new) 
    if theta_new.shape[0]==num_classes*num_l:
        theta_new = theta_new.view(num_classes,num_l).unsqueeze(0)
        a_new = a_new.view(num_classes,num_l).unsqueeze(0)
        b_new = b_new.view(num_classes,num_l).unsqueeze(0)
    else:
        # print('shape')
        theta_new = theta_new.view(-1,num_classes,num_l)
        a_new = a_new.view(-1,num_classes,num_l)
        b_new = b_new.view(-1,num_classes,num_l)  
    # print('pi',pi.shape)
    # print('theta',theta_new.shape)

    # pred = torch.zeros(y.shape).to(device)
    # for i in range(len(pred)):
    #     pred[i]=torch.matmul(pi[i],theta_new[i].view(num_classes,num_l)).view(num_l)

    pred = torch.matmul(pi.view(-1,1,num_classes),theta_new)
    # print('pred',pred.shape)
    # print('y',y.shape)
    loss = non_mse( pred.view(-1,num_l), y)
    model.deupdate_comp(model.comp_1)
    comp_1 = torch.matmul(pi.view(1,-1,num_classes).permute(0,2,1),y.view(1,-1,num_l)).view(1,num_classes*num_l)
    comp_2 = torch.matmul(pi.view(1,-1,num_classes).permute(0,2,1),(1-y).view(1,-1,num_l)).view(1,num_classes*num_l)
    comp_n = torch.cat((comp_1,comp_2),0).view(1,-1)
    model.update_comp(comp_n)
    # model.update_comp(theta.mean(0)/wnew)

    return loss

def BM_NIG_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    alpha = output[0].to(device)
    print('alpha',alpha.shape,alpha)

    Reg_loss = torch.mean((torch.sum(alpha,dim=1)-1)**2)
    # print('target',target)
    loss = torch.mean(
        nig_loss(target, output, epoch_num, num_classes, annealing_step, device=device)
    )
    pl=0.01
    loss+=pl*Reg_loss
    return loss



def BM_weiNIG_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    alpha = output[0].to(device)
    # print('alpha',alpha.shape,alpha)
    output_t = [output[0],output[2],output[3],output[4]]
    Reg_loss = torch.mean((torch.sum(alpha,dim=1)-1)**2)
    # print('target',target)
    loss = torch.mean(
        nig_loss(target, output_t, epoch_num, num_classes, annealing_step, device=device)
    )
    pl=0.01
    loss+=pl*Reg_loss
    return loss

def BM_weiNIGreg_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    alpha = output[0].to(device)
    # print('alpha',alpha.shape,alpha)
    output_t = [output[0],output[2],output[3],output[4]]
    Reg_loss = torch.mean((torch.sum(alpha,dim=1)-1)**2)
    # print('target',target)
    evid_loss = torch.mean(((alpha-target)**2)*(output[2]+0.5*output[3]+1/output[4]))
    loss = torch.mean(
        nig_loss(target, output_t, epoch_num, num_classes, annealing_step, device=device,lamb=0)
    )
    pl=0.01
    loss+=pl*Reg_loss
    loss+=0.01*evid_loss
    return loss

def BM_NON_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    alpha = output[0].to(device)
    # print('alpha',alpha.shape,alpha)
    # print('target',target)
    loss = torch.sum(
        non_mse_loss(target, output, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

def BM_sepNON_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    alpha = output[0].to(device)
    # print('alpha',alpha.shape,alpha)
    # print('target',target)
    loss = torch.mean(
        nonsep_mse_loss(target, output, epoch_num, num_classes, annealing_step, device=device,model=model,num_l=target.shape[-1])
    )
    return loss

def ML_NN_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    loss = torch.mean(
        ml_nn_loss(target, output, epoch_num, num_classes, annealing_step, device=device,model=model,num_l=target.shape[-1])
    )
    return loss

def BM_sq_loss(output, target, epoch_num, num_classes=0, \
                       annealing_step=10, device=None,model=None,num_l = 13):
    if not device:
        device = get_device()
    alpha = output[0].to(device)
    print('alpha',alpha.shape,alpha)
    print('target',target)
    # print('comp',model.comp_0.shape,model.comp_0)

    # m = nn.Softmax(dim=1)
    # pi = m(pi)
    # pi = pi.unsqueeze(0)
    theta_p = output[1].to(device)
    # print('theta',theta_p.shape,theta_p)
    a_p = theta_p[:,:num_classes*num_l]
    b_p = theta_p[:,num_classes*num_l:]
    a_sl = model.comp_0[:,:num_classes*num_l]
    b_sl = model.comp_0[:,num_classes*num_l:]
    # a_new = a_p+a_sl
    # b_new = b_p+b_sl
    a_new = a_sl
    b_new = b_sl
    theta_new = a_new/(a_new+b_new)
    # print('theta_new',theta_new.shape,theta_new)
    if theta_new.shape[0]==num_classes*num_l:
        theta_new = theta_new.view(num_classes,num_l).unsqueeze(0)
        a_new = a_new.view(num_classes,num_l).unsqueeze(0)
        b_new = b_new.view(num_classes,num_l).unsqueeze(0)
    else:
        theta_new = theta_new.view(-1,num_classes,num_l)
        a_new = a_new.view(-1,num_classes,num_l)
        b_new = b_new.view(-1,num_classes,num_l)



    # print('theta_new',theta_new.shape,theta_new)


    
    # print('theta',theta_p)
    y = target.to(device)
    if y.dim()==1:
        y=y.unsqueeze(0)

    loss = 0
    for i in range(len(theta_new)):
        S = alpha[i].sum()
        # compute first part
        Int_1 = 0
        # print('alpha[i]',alpha[i].shape,alpha[i])
        print('theta_new[i]',theta_new[i].shape,theta_new[i])

        # block=alpha[i]/S*theta_new[i]

        # print('block',block.shape,block)
        pred= torch.matmul(alpha[i]/S,theta_new[i])
        print('S',S)

        # print('y',i,y[i])
        # print('pred',pred)
        # Int_1 = Int_1+torch.sum(y[i]**2+2*y[i]*torch.sum(\
        #     torch.matmul(alpha[i]/S,theta_new[i]),dim=0))
        Int_1 = Int_1+torch.sum(y[i]**2+2*y[i]*\
            torch.matmul(alpha[i]/S,theta_new[i]))
    
        #compute second part
        Int_2 = 0
        for l in range(num_l):
            # print('block2',torch.matmul((theta_new[i,:,l]*alpha[i,:])\
            #         ,((theta_new[i,:,l]*alpha[i,:])).T))

            Int_2 = Int_2+\
                torch.sum(theta_new[i,:,l]*\
                    (a_new[i,:,l]+1)/\
                        (a_new[i,:,l]+b_new[i,:,l]+1)\
                            *alpha[i,:]*(alpha[i,:]+1)\
                                /S/(S+1))\
                +torch.sum(torch.matmul((theta_new[i,:,l]*alpha[i,:])\
                    ,((theta_new[i,:,l]*alpha[i,:])).T)/S/(S+1))
        # sum
        Int = Int_1+Int_2
        loss = loss+Int

    
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    # sq_loss = annealing_coef*loss
    y_k = torch.ones(alpha.shape).to(device)
    kl_k = int(num_classes/3)
    print('alpha',alpha.shape,alpha)
    # _,k_indices = torch.topk(alpha,kl_k,dim=1)
    # y_k[k_indices] = 0
    # kl_alpha = (alpha - 2) * y_k + 1 -(alpha )* (1-y_k)
    # # kl_alpha = alpha
    # kl_div = torch.sum(kl_alpha**2)
    # # kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    # sq_loss = loss+100*kl_div
    sq_loss = loss+0*(torch.sum((torch.ones(alpha.shape).to(device)-alpha)**2)-torch.max(alpha)**2)
    print('loss',loss)
    print('sq_loss',sq_loss)
    return sq_loss
    
def evidential_bm_loss(output, target, epoch_num, num_classes=0, num_l = 13,\
                       annealing_step=10, device=None,model=None):
    if not device:
        device = get_device()
    loss = torch.mean(
        BM_sq_loss(
             output, target, epoch_num, num_classes, \
                annealing_step, device = device,model = model,num_l = num_l,
        )
    )
    return loss

def alpha_loss(output, target, epoch_num, num_classes=0, num_l = 13,\
                       annealing_step=10, device=None,model=None):
    if not device:
        device = get_device()
    evidence = output[0].to(device)
    print('evi',evidence)
    alpha = evidence + 1
    #alpha = F.softmax(alpha1)
    print('alpha',alpha.shape,alpha)
    print('target',target) 
    # auc_macro = metrics.roc_auc_score(target.detach().cpu().numpy(),alpha.detach().cpu().numpy(),average='macro')      
    # print('auc_macro',auc_macro)
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    # loss = torch.mean(
    #     edl_loss(
    #         torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
    #     )
    # )
    # loss = torch.mean(
    #     BM_sq_loss(
    #          output, target, epoch_num, num_classes, \
    #             annealing_step, device = device,model = model,num_l = num_l,
    #     )
    # )
    return loss