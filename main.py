import numpy as np
from edl_pytorch import NormalInvGamma, evidential_regression
import copy
import time
import torch
import csv
from model import *
from BM import *
from datasets import *
from loss import *
from util import *
from config import get_args
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def get_device():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	return device

def train_sep_bm(    model,
	dataloaders,
	num_classes,
	criterion,
	optimizer,
	scheduler=None,
	num_epochs=25,
	device=None,
	uncertainty=False,
	bookkeep = False,
	bk_w = 1,
	num_l = 13, fname='1'
):
	since = time.time()

	if not device:
		device = get_device()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	loss_List = []
	with torch.autograd.set_detect_anomaly(True):
		for epoch in range(num_epochs):
			print("Epoch {}/{}".format(epoch, num_epochs - 1))
			print("-" * 10)
			for phase in ["train", "val"]:
				if phase == "train":
					print("Training...")
					model.train()  # Set model to training mode
				else:
					print("Validating...")
					model.eval()  # Set model to evaluate mode

				running_loss = 0.0
				running_corrects = 0.0
				correct = 0

				# Iterate over data.
				comp_sum = 0

				
				for i, (inputs, labels) in enumerate(dataloaders[phase]):

					inputs = inputs.to(device)
					labels = labels.to(device)

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == "train"):

						y = labels.to(device)
						# print('inputs',inputs)
						# print('labels',labels)
						outputs = model(inputs)
						# print('outputs',outputs)
						if bookkeep:
							theta_s = outputs[1].to(device)
							# print('theta_keep',theta_s.shape,theta_s)
							theta_s = theta_s.sum(dim=0).view(-1,2*num_classes*num_l)
							comp_sum+=theta_s.detach().cpu().numpy()

						loss = criterion(outputs, y.float(), epoch,\
							num_classes=num_classes, num_l = num_l,\
					   annealing_step=10, device=device,model=model
						)

						if phase == "train":
							loss.backward()
							torch.nn.utils.clip_grad_value_(model.parameters(), 100)
							optimizer.step()

					# statistics
					running_loss += loss.item() * inputs.size(0)


				if scheduler is not None:
					if phase == "train":
						scheduler.step()

				epoch_loss = running_loss / len(dataloaders[phase].dataset)
				print('epoch loss',epoch_loss)
				loss_List.append(epoch_loss)
				if bookkeep:
					model.update_comp(comp_sum*bk_w/ len(dataloaders[phase].dataset))

			print()

	time_elapsed = time.time() - since
	print(
		"Training complete in {:.0f}m {:.0f}s".format(
			time_elapsed // 60, time_elapsed % 60
		)
	)
	print("Best val Acc: {:4f}".format(best_acc))

	comp_save = model.comp_0.cpu().detach().numpy()
	print('comp_save',comp_save)
	file_p = open('./save_path'+fname+'comp_save_2.npy','wb')
	np.save(file_p,comp_save)
	file_p = open('./save_path'+fname+'mla_loss_2.npy','wb')
	np.save(file_p,loss_List)
	return model,loss_List    

def main():
	args,SEED = get_args()

	fname = 'date'+args.ds+args.fname
	fnamesub = fname+'.csv'
	header = ['step','macro_auc','micro_auc']
	#replace the save path
	with open('./save_path'+fnamesub, 'w') as f:
		writer_obj = csv.writer(f)
		writer_obj.writerow(header)

	wnew = args.wnew

	x,y=readDataset(args.ds)
	print(x.shape)
	train = args.train
	pool = args.pool
	test = args.test
	deterministic(SEED)
	train_index,candidate_index,test_index=split(x,label=y,train_rate=train,\
		candidate_rate=pool,test_rate=test,\
			seed=SEED,even=True)
	print('train',len(train_index))
	print('test',len(test_index))
	print('pool',len(candidate_index))
	loss_all = []
	if (args.bookkeep == 'sepopt'):
		bk = True
	else:
		bk= False
	for iter_al in range(args.AL_rounds):
		x_train = x[train_index]
		x_test = x[test_index]
		y_train = y[train_index]
		y_test = y[test_index]
		x_can = x[candidate_index]
		y_can = y[candidate_index]


		components_to_test = args.n_components

		model = BernoulliMixture(components_to_test, 200)
		
		drop_p=0.1
		model.fit(y_train)
		score = model.score(y_test)
		
		
		cluster_ids = np.arange(0, components_to_test)
		
		pi = model.pi
		
		mu = model.mu

		mu_a = np.concatenate([model.mu,1-model.mu]) 

		comp0 = mu_a.reshape(1,-1)*len(x_train)

		num_classes = components_to_test
		alpha_t = np.ones((num_classes,len(x)))
		for j in range(len(x)):
			xsol2,lossres=opt_alpha(mu,x[j,:],y[j,:],num_classes)
			alpha_t[:,j]=xsol2
		# print('alpha_t',alpha_t)
		# print('mu',mu)
		alpha_t = alpha_t.T

		alpha_train = alpha_t[train_index]
		alpha_test = alpha_t[test_index]
		alphatrain = torch.tensor(alpha_train)
		alphatest = torch.tensor(alpha_test)
		xtrain = torch.tensor(x_train)
		xtest = torch.tensor(x_test)
		ytrain = torch.tensor(y_train)
		ytest = torch.tensor(y_test)
		print('ytest',ytest.shape)
		ycan = torch.tensor(y_can)
		xcan = torch.tensor(x_can)

		# from sklearn.kernel_ridge import KernelRidge
		# krr = KernelRidge(kernel='rbf',alpha=0.10)
		# from sklearn.metrics import mean_squared_error
		# krr.fit(x_train,alpha_train)

		# None = krr.predict(x)




		trainData=myDataset(xtrain.to(device),alphatrain.to(device))
		testData=myDataset(xtest.to(device),alphatest.to(device))
		train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
		test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
		dataloaders = {
		"train": train_dataloader,
		"val": test_dataloader,
	}   

		num_l = y.shape[1]
		out_size = num_l
		num_feature = x.shape[1]
		in_size = num_feature
		bm_model = BM_sep_EDR_NN_custom(in_size = in_size,\
			hidden_size = args.m_hidden,out_size = out_size,\
			n_comp=num_classes,embed = args.m_embed,\
			cons=args.m_cons,\
			drop_p = args.m_drop_p, activation=args.m_activation,\
			decoder_type = args.m_decoder_type).to(device)

		pytorch_total_params = sum(p.numel() for p in bm_model.parameters())

		print(" Number of Parameters: ", pytorch_total_params)

		if (args.optimizer=='adam'):
			optimizer = optim.Adam(bm_model.parameters(), lr=args.lr, weight_decay=args.wd)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)




		#weights pretraining
		if (args.pretrain_loss=='NIG'):
			criterion = BM_weiNIG_loss
		elif (args.pretrain_loss=='NIGreg'):
			criterion = BM_weiNIGreg_loss


		
		bm_model.update_comp(comp0)
		
		model_opt,loss_opt = train_sep_bm(    bm_model,
		dataloaders,
		num_classes,
		criterion,
		optimizer,
		scheduler=scheduler,
		num_epochs=args.pretrain_epochs,
		device=None,
		uncertainty=False,
		bookkeep=False,
		num_l = num_l,
		fname=fname
		)
		model_opt.eval()
		loss_all+=loss_opt

		datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
			  alpha_t, num_l,wnew,n=0,\
				alpha_test=alpha_test, alpha_train= alpha_train,\
				train_index=train_index,test_index=test_index,\
				handle='train_test',pi=None,bs_pred = None,\
					ysum=False,\
				xtest=xtest,ytest=ytest,\
					x_train=x_train,y_train=y_train)
		with open('./save_path'+fnamesub,'a') as f:
			writer_obj = csv.writer(f)
			writer_obj.writerow(datapoint)

		for iter in range(args.tr_rounds):
			trainData=myDataset(xtrain.to(device),ytrain.to(device))
			testData=myDataset(xtest.to(device),ytest.to(device))
			train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
			test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
			dataloaders = {
			"train": train_dataloader,
			"val": test_dataloader,
		}   

			bm_model = model_opt.to(device)

			optimizer = optim.Adam(bm_model.decoder2.parameters(), \
								   lr=1e-3, weight_decay=0)

			if(args.l_loss=='NON'):
				criterion = BM_sepNON_loss
			model_opt,loss_opt = train_sep_bm(    bm_model,
			dataloaders,
			num_classes,
			criterion,
			optimizer,
			scheduler=scheduler,
			num_epochs=args.l_epochs,
			device=None,
			uncertainty=False,
			bookkeep=bk,
			bk_w = args.bk_w,
			num_l = num_l,
			fname=fname
			)
			loss_all+=loss_opt
			model_opt.eval()

			datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
				alpha_t, num_l,wnew,n=iter,\
					alpha_test=alpha_test, alpha_train= alpha_train,\
					train_index=train_index,test_index=test_index,\
					handle='train_test',pi=None,bs_pred = None,\
						ysum=False,\
					xtest=xtest,ytest=ytest,\
						x_train=x_train,y_train=y_train)
			with open('./save_path'+fnamesub,'a') as f:
				writer_obj = csv.writer(f)
				writer_obj.writerow(datapoint)

# MSM for weights step
			if args.msm_step:
				trainData=myDataset(xtrain.to(device),ytrain.to(device))
				testData=myDataset(xtest.to(device),ytest.to(device))
				train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
				test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
				dataloaders = {
				"train": train_dataloader,
				"val": test_dataloader,
			}   

				bm_model = model_opt.to(device)

				optimizer = optim.Adam(bm_model.encoder.parameters(), \
									lr=1e-3, weight_decay=0)
				optimizer.add_param_group({'params': bm_model.decoder1.parameters()})
				if(args.l_loss=='NON'):
					criterion = BM_sepNON_loss
				model_opt,loss_opt = train_sep_bm(    bm_model,
				dataloaders,
				num_classes,
				criterion,
				optimizer,
				scheduler=scheduler,
				num_epochs=args.msm_epochs,
				device=None,
				uncertainty=False,
				bookkeep=False,
				bk_w = args.bk_w,
				num_l = num_l,
				fname=fname
				)
				loss_all+=loss_opt
				model_opt.eval()

				datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
					alpha_t, num_l,wnew,n=iter,\
						alpha_test=alpha_test, alpha_train= alpha_train,\
						train_index=train_index,test_index=test_index,\
						handle='train_test',pi=None,bs_pred = None,\
							ysum=False,\
						xtest=xtest,ytest=ytest,\
							x_train=x_train,y_train=y_train)
				with open('./save_path'+fnamesub,'a') as f:
					writer_obj = csv.writer(f)
					writer_obj.writerow(datapoint)


			if args.opt_mu:
				print('optimize mu!', args.opt_mu)
				a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
				b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
				mu = a_sl/(a_sl+b_sl)
				mu = mu.reshape(num_classes,num_l)
				alpha_t = np.ones((num_classes,len(x)))
				for j in range(len(x)):
					xsol2,lossres=opt_alpha(mu,x[j,:],y[j,:],num_classes)
					alpha_t[:,j]=xsol2
				alpha_t = alpha_t.T                
				alpha_train = alpha_t[train_index]
				alpha_test = alpha_t[test_index]
				alphatrain = torch.tensor(alpha_train)
				alphatest = torch.tensor(alpha_test)

			trainData=myDataset(xtrain.to(device),alphatrain.to(device))
			testData=myDataset(xtest.to(device),alphatest.to(device))
			train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
			test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
			dataloaders = {
			"train": train_dataloader,
			"val": test_dataloader,
		}   

			bm_model = model_opt.to(device)

			pytorch_total_params = sum(p.numel() for p in bm_model.parameters())

			print(" Number of Parameters: ", pytorch_total_params)

			optimizer = optim.Adam(bm_model.parameters(), lr=1e-3, weight_decay=0)

			if(args.pi_loss=='NIG'):
				criterion = BM_weiNIG_loss
			model_opt,loss_opt = train_sep_bm(    bm_model,
			dataloaders,
			num_classes,
			criterion,
			optimizer,
			scheduler=scheduler,
			num_epochs=args.pi_epochs,
			device=None,
			uncertainty=False,
			bookkeep=False,
			num_l = num_l,
			fname=fname
			)
			loss_all+=loss_opt
			model_opt.eval()

			datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
				alpha_t, num_l,wnew,n=iter,\
					alpha_test=alpha_test, alpha_train= alpha_train,\
					train_index=train_index,test_index=test_index,\
					handle='train_test',pi=None,bs_pred = None,\
						ysum=False,\
					xtest=xtest,ytest=ytest,\
						x_train=x_train,y_train=y_train)
			with open('./save_path'+fnamesub,'a') as f:
				writer_obj = csv.writer(f)
				writer_obj.writerow(datapoint)


# update all data and train baseline
		a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
		b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()


		comp0 = a_sl/(a_sl+b_sl)

		num_classes = components_to_test
		alpha_t = np.ones((num_classes,len(x)))
		for j in range(len(x)):
			xsol2,lossres=opt_alpha(mu,x[j,:],y[j,:],num_classes)
			alpha_t[:,j]=xsol2
		# print('alpha_t',alpha_t)
		# print('mu',mu)
		alpha_t = alpha_t.T


		alpha_train = alpha_t[train_index]
		alpha_test = alpha_t[test_index]

		alphatrain = torch.tensor(alpha_train)
		alphatest = torch.tensor(alpha_test)
		xtrain = torch.tensor(x_train)
		xtest = torch.tensor(x_test)
		ytrain = torch.tensor(y_train)
		ytest = torch.tensor(y_test)
		print('ytest',ytest.shape)
		ycan = torch.tensor(y_can)
		xcan = torch.tensor(x_can)


		trainData=myDataset(xtrain.to(device),alphatrain.to(device))
		testData=myDataset(xtest.to(device),alphatest.to(device))
		train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
		test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
		dataloaders = {
		"train": train_dataloader,
		"val": test_dataloader,
	} 
		output_t = model_opt(xcan.float().to(device))

		mu_can = np.array(output_t[0].cpu().detach().numpy())
		v2_can = np.array(output_t[2].cpu().detach().numpy())
		a2_can = np.array(output_t[3].cpu().detach().numpy())
		b2_can = np.array(output_t[4].cpu().detach().numpy())

		met = args.al_mtd
		batch_size = args.AL_batch

		if met =='bvs':
			probs_sorted,idxs = output_t[0].sort(descending=True)
			U = probs_sorted[:,0]-probs_sorted[:,1]
			ind_add = list(np.array(candidate_index)[list(U.sort()[1][:batch_size].cpu().numpy())])
			train_index = train_index+ind_add
			for ind_a in ind_add:
				candidate_index.remove(ind_a)
		if met =='entropy':
			theta_p = output_t[1].detach().cpu().numpy()
			a_p = theta_p[:,:num_classes*num_l]
			b_p = theta_p[:,num_classes*num_l:]
			a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
			b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
			a_sl = np.repeat(a_sl,len(a_p),axis=0)
			b_sl = np.repeat(b_sl,len(b_p),axis=0)

			a_new = a_p/wnew+a_sl
			b_new = b_p/wnew+b_sl

			theta_old = a_sl/(a_sl+b_sl)
			theta_new = a_new/(a_new+b_new)


			pred = np.zeros((len(y_can),y_can.shape[-1]))
			for i in range(len(y_can)):
				pred_0 = np.matmul(mu_can[i],theta_new[i].reshape(components_to_test,-1))
				print('pred_0',pred_0.shape)
				pred[i] = pred_0 
			print('predshape',pred.shape)
			ent = np.sum(pred*np.log(pred),axis=1)
			U = ent
			ind_add = list(np.array(candidate_index)[list(np.argsort(U)[:batch_size])])
			train_index = train_index+ind_add
			for ind_a in ind_add:
				candidate_index.remove(ind_a)



		elif met =='evidential_sample':
			np.save('./save_path'+fname+'probs.npy',output_t[0].detach().cpu().numpy())
			theta_p = output_t[1].detach().cpu().numpy()
			a_p = theta_p[:,:num_classes*num_l]
			b_p = theta_p[:,num_classes*num_l:]
			a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
			b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
			a_sl = np.repeat(a_sl,len(a_p),axis=0)
			b_sl = np.repeat(b_sl,len(b_p),axis=0)

			a_new = a_p/wnew+a_sl
			b_new = b_p/wnew+b_sl

			theta_old = a_sl/(a_sl+b_sl)
			theta_new = a_new/(a_new+b_new)

			dif_theta = np.diag(cosine_similarity(theta_old,theta_new))

			pred = np.zeros((len(y_can),y_can.shape[-1]))
			cov1 = np.zeros((len(y_can)))
			cov2 = np.zeros((len(y_can),y_can.shape[-1],y_can.shape[-1]))
			for i in range(len(y_can)):
				pred_0 = np.matmul(mu_can[i],theta_new[i].reshape(components_to_test,-1))
				print('pred_0',pred_0.shape)
				pred[i] = pred_0 
				covm = model_opt.compute_cov()
				# print('pis',output_t[0].cpu().detach().numpy())

				for k in range(components_to_test):
					# print('pis',output_t[0].cpu().detach().numpy()[k])
					# cov2[i]+=output_t[0].cpu().detach().numpy()[:,k]*covm[k].cpu().detach().numpy()
					cov2[i]+=mu_can[i][k]*covm[k].cpu().detach().numpy()

				cov2[i]-=np.matmul(pred_0,pred_0.T)
				cov1[i]=np.linalg.det(cov2[i])

			cov = np.matmul(pred,pred.T)

			evid = v2_can+0.5*a2_can+1/b2_can
			probs = np.mean(evid,axis=1)+args.s_lambda*(cov1)-args.s_eta*dif_theta


			ind_add = list(np.array(candidate_index)[list(np.argsort(probs)[::-1][:batch_size])])
			train_index = train_index+ind_add
			print(ind_add)
			for ind_a in ind_add:
				candidate_index.remove(ind_a)


		elif met =='cvirs':
			np.save('./save_path'+fname+'probs.npy',output_t[0].detach().cpu().numpy())
			theta_p = output_t[1].detach().cpu().numpy()
			a_p = theta_p[:,:num_classes*num_l]
			b_p = theta_p[:,num_classes*num_l:]
			a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
			b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
			a_sl = np.repeat(a_sl,len(a_p),axis=0)
			b_sl = np.repeat(b_sl,len(b_p),axis=0)

			a_new = a_p/wnew+a_sl
			b_new = b_p/wnew+b_sl

			theta_old = a_sl/(a_sl+b_sl)
			theta_new = a_new/(a_new+b_new)


			pred = np.zeros((len(y_can),y_can.shape[-1]))
			for i in range(len(y_can)):
				pred_0 = np.matmul(mu_can[i],theta_new[i].reshape(components_to_test,-1))
				print('pred_0',pred_0.shape)
				pred[i] = pred_0 
			print('predshape',pred.shape)
			ind_add = cvirs_Sample_emlc( y, train_index, candidate_index, test_index, pred,batch_size)
			train_index = train_index+ind_add
			print(ind_add)
			for ind_a in ind_add:
				candidate_index.remove(ind_a)

		np.save('./save_path'+fname+'loss_all.npy',loss_all)


if __name__ == "__main__":
	main()    