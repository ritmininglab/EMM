import torch
import random
import heapq
import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def deterministic(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	# torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def print_res(model_opt, x_test, y_test, mu, fname,num_classes,
			  ground_truth, num_l,wnew,n=0,\
				alpha_test=None, alpha_train= None,\
				train_index=None,test_index=None,\
				handle='train_test',pi=None,bs_pred = None,ysum=False,\
				xtest=None,ytest=None,x_train=None,y_train=None):

	components_to_test = num_classes
	mu2 = np.array(mu)

	if (alpha_test is None):
		alpha_test = ground_truth[test_index]
		alpha_train = ground_truth[train_index]

	if(handle=='train_test'): 
		xtrain = torch.tensor(x_train)
		ytrain = torch.tensor(y_train)
	if(xtest is None):
		xtest = torch.tensor(x_test)
		ytest = torch.tensor(y_test)
		if(handle=='train_test'): 
			xtrain = torch.tensor(x_train)
			ytrain = torch.tensor(y_train)
	
	output_t = model_opt(xtest.float().to(device))

	al2 = np.array(output_t[0].cpu().detach().numpy())



	al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
	pred2 = np.matmul(al2p,mu2)

	y1=y_test

	al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
	pred2 = np.matmul(al2p,mu2)
	# y1 = np.array(ytest)
	y1=y_test
	y1s = np.sum(y1,0)
	col = np.where(y1s!=0)[0]
	y2 = y1[:,col]



	if (bs_pred is not None):
		al_train = bs_pred[train_index]
		al_test = bs_pred[test_index]
		pred3 = np.matmul(al_test,mu2)
		pred32 = pred3[:,col]
		test_auc_micro_3 = metrics.roc_auc_score(y2,pred32,average='micro')
		test_auc_macro_3 = metrics.roc_auc_score(y2,pred32,average='macro')
		print('test mi:alpha3',test_auc_micro_3)
		print('test ma:alpha3',test_auc_macro_3)

	theta_p = output_t[1].detach().cpu().numpy()
	# print('theta',theta_p.shape)
	a_p = theta_p[:,:num_classes*num_l]
	b_p = theta_p[:,num_classes*num_l:]
	# print('comp_0',model_opt.comp_0, model_opt.comp_0.shape)
	a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
	b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
	a_sl = np.repeat(a_sl,len(a_p),axis=0)
	b_sl = np.repeat(b_sl,len(b_p),axis=0)

	a_new = a_p/wnew+a_sl
	b_new = b_p/wnew+b_sl

	theta_new = a_new/(a_new+b_new)
	# print('al2p',al2p.shape)
	pred = np.zeros((len(y_test),y_test.shape[-1]))
	for i in range(len(y_test)):
		pred_0 = np.matmul(al2p[i],theta_new[i].reshape(components_to_test,-1))
		pred[i] = pred_0 

	auc_micro10 = metrics.roc_auc_score(y2,pred[:,col],average='micro')


	auc_macro10 = metrics.roc_auc_score(y2,pred[:,col],average='macro')



# training results 

	al1 = np.array(alpha_train)
	output_t = model_opt(xtrain.float().to(device))

	al2 = np.array(output_t[0].cpu().detach().numpy())




	al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)

	y1=y_train
	y1s = np.sum(y1,0)
	col = np.where(y1s!=0)[0]
	y2 = y1[:,col]



	if (bs_pred is not None):
		mse1 = mean_squared_error(alpha_train,al_train)
		mse4 = mean_squared_error(alpha_test,al_test)

		pred3 = np.matmul(al_train,mu2)
		pred32 = pred3[:,col]

		auc_micro_3 = metrics.roc_auc_score(y2,pred32,average='micro')
		auc_macro_3 = metrics.roc_auc_score(y2,pred32,average='macro')
		print('mi:alpha3',auc_micro_3)
		print('ma:alpha3',auc_macro_3)




	if (bs_pred is not None):
		datapoint = [n, auc_macro_3,test_auc_macro_3,auc_macro10,test_auc_micro_3,auc_micro10]
	else:
		datapoint = [n,auc_macro10,auc_micro10]


	return datapoint

def prediction(model_opt, x_test, y_test, mu, fname,num_classes,
			  ground_truth, num_l,wnew,n=0,\
				alpha_test=None, alpha_train= None,\
				train_index=None,test_index=None,\
				handle='train_test',pi=None,bs_pred = None,ysum=False,\
				xtest=None,ytest=None,x_train=None,y_train=None):
	mse1 = 0
	mse4 = 0
	components_to_test = num_classes
	mu2 = np.array(mu)

	if (alpha_test is None):
		alpha_test = ground_truth[test_index]
		alpha_train = ground_truth[train_index]

	if(handle=='train_test'): 
		xtrain = torch.tensor(x_train)
		ytrain = torch.tensor(y_train)
	if(xtest is None):
		xtest = torch.tensor(x_test)
		ytest = torch.tensor(y_test)
		if(handle=='train_test'): 
			xtrain = torch.tensor(x_train)
			ytrain = torch.tensor(y_train)
	
	al1 = np.array(alpha_test)
	output_t = model_opt(xtest.float().to(device))

	al2 = np.array(output_t[0].cpu().detach().numpy())

	#al2 = al2+1
	# al1 = al1.T
	pred1 = np.matmul(al1,mu2)
	al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
	pred2 = np.matmul(al2p,mu2)
	# y1 = np.array(ytest)
	y1=y_test
	mse2 = mean_squared_error(al1,al2)
	mse22 = mean_squared_error(alpha_test,al2)

	# if ysum:
	#     y1s = np.sum(y1,0)
	#     col = np.where(y1s!=0)[0]
	#     # print('sum',y1s[col])
	#     y2 = y1[:,col]
	#     pred12 = pred1[:,col]
	#     pred22 = pred2[:,col]
	#     mse5 = mean_squared_error(al1,alpha_test)
	# else:
	#     y2=y1
	#     pred12 = pred1
	#     pred22 = pred2




	pred1 = np.matmul(alpha_test,mu2)

	al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
	pred2 = np.matmul(al2p,mu2)
	# y1 = np.array(ytest)
	y1=y_test
	y1s = np.sum(y1,0)
	col = np.where(y1s!=0)[0]
	y2 = y1[:,col]
	pred12 = pred1[:,col]
	pred22 = pred2[:,col]



	theta_p = output_t[1].detach().cpu().numpy()
	print('theta',theta_p.shape)
	a_p = theta_p[:,:num_classes*num_l]
	b_p = theta_p[:,num_classes*num_l:]
	print('comp_0',model_opt.comp_0, model_opt.comp_0.shape)
	a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
	b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
	a_sl = np.repeat(a_sl,len(a_p),axis=0)
	b_sl = np.repeat(b_sl,len(b_p),axis=0)

	a_new = a_p/wnew+a_sl
	b_new = b_p/wnew+b_sl

	theta_new = a_new/(a_new+b_new)
	print('al2p',al2p.shape)
	pred = np.zeros((len(y_test),y_test.shape[-1]))
	for i in range(len(y_test)):
		pred_0 = np.matmul(al2p[i],theta_new[i].reshape(components_to_test,-1))
		pred[i] = pred_0 

	return y2,pred22, pred[:,col]

def fcv(Yi,Yj):
	q = len(Yi)
	Yi_1 = Yi[Yj==1]
	a = np.sum(Yi_1)
	c = len(Yi_1)-a
	Yi_0 = Yi[Yj==0]
	b = np.sum(Yi_0)
	d = len(Yi_0)-b
	DH = (b+c)/q
	# print( 'a = '+str(a)+'b = '+str(b)+'c = '+str(c)+'d = '+str(d) )

	if DH<1 :
		H21 = -DH*np.log(DH)-(a+d)/q*np.log((a+d)/q)
		H22 = -b/(b+c)*np.log(b/(b+c))-c/(b+c)*np.log(c/(b+c))
		H23 = -a/(a+d)*np.log(a/(a+d))-d/(a+d)*np.log(d/(a+d))
		
		HYY = H21+(b+c)/q*H22+(a+d)/q*H23
		wi = len(Yi_1)
		wj = len(Yj[Yj==1])
		HYi = -wi/q*np.log(wi/q)-(q-wi)/q*np.log((q-wi)/q)
		HYj = -wj/q*np.log(wj/q)-(q-wj)/q*np.log((q-wj)/q)
	
		DE = 2-(HYi+HYj)/HYY

	else:
		DE = 1
		
	if math.isnan(DE):
		DE = 0
	return DE

def cvirs_Sample(data, Y, train_index, candidate_index, test_index, clf):
	# self.data = data
	# self.Y = Y
	# self.train_index = train_index
	# self.candidate_index = candidate_index
	# self.test_index = test_index
	pUN = clf.predict_proba(data[candidate_index,:])
	mIL = 2*pUN-1
	s_List = []
	U_s = len(candidate_index)
	q = pUN.shape[1]
	
	v_List = []
	# yUN = clf.predict(X[candidate_index,:])
	yUN = np.zeros(pUN.shape)

	yUN[pUN>0.5] = 1
	yL = Y[train_index,:]
	
	L_s = len(train_index)
	resultList = []
	for i in range(len(candidate_index)):
		# compute s
		S_up = 0
		for j in range(q):
			mIL_col = mIL[:,j]
			tau = np.zeros(U_s)
			tau[mIL_col<mIL[i,j]] = 1
			S_up = S_up + U_s - np.sum(tau)
		S = S_up/(q*(U_s-1))
		s_List.append(S)
		# compute v
		V = 0
		y_col = yUN[i,:]
		for k in range(L_s):
			fYY = fcv(y_col,yL[k,:])
			# print(fYY)
			V = V+fYY
		V = V/L_s
		v_List.append(V)
		resultList.append(S*V)

	
	# targetIndex = resultList.index( min(resultList) )
	targetIndex = list(map(resultList.index,heapq.nlargest(5, resultList ) ));
	return targetIndex,resultList,s_List,v_List


def cvirs_Sample_emlc( Y, train_index, candidate_index, test_index, pred, batch_size=5):
	# self.data = data
	# self.Y = Y
	# self.train_index = train_index
	# self.candidate_index = candidate_index
	# self.test_index = test_index
	pUN = pred
	mIL = 2*pUN-1
	s_List = []
	U_s = len(candidate_index)
	q = pUN.shape[1]
	
	v_List = []
	# yUN = clf.predict(X[candidate_index,:])
	yUN = np.zeros(pUN.shape)

	yUN[pUN>0.5] = 1
	yL = Y[train_index,:]
	
	L_s = len(train_index)
	resultList = []
	for i in range(len(candidate_index)):
		# compute s
		S_up = 0
		for j in range(q):
			mIL_col = mIL[:,j]
			tau = np.zeros(U_s)
			tau[mIL_col<mIL[i,j]] = 1
			S_up = S_up + U_s - np.sum(tau)
		S = S_up/(q*(U_s-1))
		s_List.append(S)
		# compute v
		V = 0
		y_col = yUN[i,:]
		for k in range(L_s):
			fYY = fcv(y_col,yL[k,:])
			# print(fYY)
			V = V+fYY
		V = V/L_s
		v_List.append(V)
		resultList.append(S*V)

	
	# targetIndex = resultList.index( min(resultList) )
	# targetIndex = list(map(resultList.index,heapq.nlargest(5, resultList ) ));
	targetIndex = list(np.array(candidate_index)[list(np.argsort(resultList)[::-1][:batch_size])])
	return targetIndex