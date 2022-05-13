import copy
import random
import pickle
import torch
import sys
import os
random.seed(10)


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_sysmdl import SystemModel
from Extended_data import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from Extended_KalmanNet_nn import KalmanNetNN
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF, bcolors
from Plot import Plot_extended as Plot
from datetime import datetime
from Logger import Logger
from filing_paths import path_model
sys.path.insert(1, path_model)
from Simulations.Lorenz_Atractor.parameters import T, T_test, m1x_0, m2x_0, m, n, delta_t_gen, delta_t
from Simulations.Lorenz_Atractor.model import f, h, fInacc, hInacc, fRotate, h_nonlinear

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")


################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%d.%m.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
unsupervised_weight = 0.0
logger = Logger(strTime, "Logs", "KalmanNet", unsupervised_weight)
logger.logEntry("Current Time = " + strTime)
print(bcolors.UNDERLINE + bcolors.BOLD + "Current Time = " + strNow + bcolors.ENDC)

######################################
###  Compare EKF, RTS and RTSNet   ###
######################################
offset = 0
chop = False
DatafolderName = 'Simulations/Lorenz_Atractor/data/T200' + '/'
# data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=dev)
# [true_sequence] = data_gen_file['All Data']

r2 = torch.tensor([1])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2.to(torch.double))
vdB = -20  # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v, r2)
q = torch.sqrt(q2.to(torch.double))

### q and r optimized for EKF
r2optdB = torch.tensor([3.0103])
ropt = torch.sqrt(10**(-r2optdB/10))
q2optdB = torch.tensor([18.2391])
qopt = torch.sqrt(10**(-q2optdB/10))

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['data_lor_v20_rq020_T200.pt']  #,'data_lor_v20_r1e-2_T100.pt','data_lor_v20_r1e-3_T100.pt','data_lor_v20_r1e-4_T100.pt']
# EKFResultName = 'EKF_nonLinearh_rq00_T20' 

for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
   #Model
   sys_model = SystemModel(f, q[rindex], h, r[rindex], T, T_test, m, n, "Lor")
   sys_model.InitSequence(m1x_0, m2x_0)

   sys_model_partialf = SystemModel(fInacc, q[rindex], h, r[rindex], T, T_test, m, n, "Lor")
   sys_model_partialf.InitSequence(m1x_0, m2x_0)

   sys_model_partialf_optq = SystemModel(fInacc, qopt, h, r[rindex], T, T_test, m, n, 'lor')
   sys_model_partialf_optq.InitSequence(m1x_0, m2x_0)

   sys_model_partialh = SystemModel(f, q[rindex], h_nonlinear, r[rindex], T, T_test, m, n, "Lor")
   sys_model_partialh.InitSequence(m1x_0, m2x_0)

   sys_model_partialh_optr = SystemModel(f, q[rindex], h_nonlinear, ropt, T, T_test, m, n, 'lor')
   sys_model_partialh_optr.InitSequence(m1x_0, m2x_0)
   
   ### Generate and load data DT case
   # print("Start Data Gen")
   # DataGen(sys_model, DatafolderName + dataFileName[0], T, T_test,randomInit=False)
   print(bcolors.UNDERLINE + bcolors.BOLD + "Loading Data: " + dataFileName[0] + bcolors.ENDC)
   [train_input_long,train_target_long, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName + dataFileName[0], map_location=dev)

   if chop:
      print("chop training data")    
      [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
      # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, T)
   else:
      print("no chopping") 
      train_target = train_target_long[:, :, 0:T]
      train_input = train_input_long[:, :, 0:T]
      # cv_target = cv_target[:,:,0:T]
      # cv_input = cv_input[:,:,0:T]  

   print("Train dataset size:", train_target.size())
   print("CV dataset size:", cv_target.size())
   print("Test dataset size:", test_target.size())

   logger.logEntry("Train dataset size:" + str(train_target.size()))
   logger.logEntry("CV dataset size:" + str(cv_target.size()))
   logger.logEntry("Test dataset size:" + str(test_target.size()))

   
   """
   ### Generate and load Lorenz Attractor data for decimation case (chopped)
   print("Data Gen")
   [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   print(test_target.size())
   [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)

   [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   [cv_target, cv_input] = Short_Traj_Split(cv_target_long, cv_input_long, T)
   """
   run_EKF_models = False

   if run_EKF_models:
      #Evaluate EKF true
      strModel = "Evaluate EKF true"
      start_time = datetime.now()
      print(strModel)
      [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
      end_time = datetime.now()
      print("Elapsed time " + strModel + " = ", end_time - start_time)

      #Evaluate EKF partial (h or r)
      strModel = "Evaluate EKF partial"
      start_time = datetime.now()
      print(strModel)
      [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialh, test_input, test_target)
      end_time = datetime.now()
      print("Elapsed time " + strModel + " = ", end_time - start_time)

      #Evaluate EKF partial optq
      strModel = "Evaluate EKF partial optimal Q"
      start_time = datetime.now()
      print(strModel)
      [MSE_EKF_linear_arr_partialoptq, MSE_EKF_linear_avg_partialoptq, MSE_EKF_dB_avg_partialoptq, EKF_KG_array_partialoptq, EKF_out_partialoptq] = EKFTest(sys_model_partialf_optq, test_input, test_target)
      end_time = datetime.now()
      print("Elapsed time " + strModel + " = ", end_time - start_time)

      #Evaluate EKF partial optr
      strModel = "Evaluate EKF partial optimal R"
      start_time = datetime.now()
      print(strModel)
      [MSE_EKF_linear_arr_partialoptr, MSE_EKF_linear_avg_partialoptr, MSE_EKF_dB_avg_partialoptr, EKF_KG_array_partialoptr, EKF_out_partialoptr] = EKFTest(sys_model_partialh_optr, test_input, test_target)
      end_time = datetime.now()
      print("Elapsed time " + strModel + " = ", end_time - start_time)


   # KNet without model mismatch
   print(bcolors.HEADER + "KNet with full model info" + bcolors.ENDC)
   modelFolder = 'KNet' + '/'
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   logger.logEntry("KNet with full model info")

   KNet_Pipeline.setssModel(sys_model)
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model)
   KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=10, learningRate=1e-3, weightDecay=1e-4) # Original Settings
   numEpochs = 1000 # actually, this parameter is not the number of Epochs, but rather the number of examples in training.
   numBatches = 10

   logger.logEntry("Unsupervised Weight = " + str(unsupervised_weight))
   print(bcolors.UNDERLINE + bcolors.BOLD + "Unsupervised Weight = " + str(unsupervised_weight) + bcolors.ENDC)
   KNet_Pipeline.setTrainingParams(n_Epochs=numEpochs, n_Batch=numBatches, learningRate=5e-4, weightDecay=1e-4)

   # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")


# make a loop of size of data to train.
   # for num in range(10, train_input.size(0)+1, int(train_input.size(0) / 100)):
   for num in range(100, 1001, 100):
      num_labeled_examples = int(num)
      logger.set_num_labeled_examples(num_labeled_examples)
      num_test_examples = N_T       #int(num_examples / 5)
      num_cross_validation = N_CV   #int(num_examples / 10)

      # Getting back the objects:
      pkl_file_name = 'objs.pkl'
      delete_flag = False
      if os.path.isfile(pkl_file_name):
         if delete_flag:
            os.remove(pkl_file_name)
            KNet_Pipeline_1st = copy.deepcopy(KNet_Pipeline)
            # Saving the objects before a successful run:
            with open(pkl_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump([KNet_Pipeline_1st, train_input, train_target, cv_input, cv_target, test_input, test_target], f)
         else:
            with open(pkl_file_name, 'rb') as f:
               del(KNet_Pipeline)
               KNet_Pipeline, train_input, train_target, cv_input, cv_target, test_input, test_target = pickle.load(f)
               KNet_Pipeline.setTrainingParams(n_Epochs=numEpochs, n_Batch=numBatches, learningRate=5e-4, weightDecay=1e-4)
               # train_input2 = torch.index_select(train_input, 0, torch.tensor(range(num_examples)))

      KNet_Pipeline.NNTrain(N_E, num_labeled_examples, train_input, train_target, num_cross_validation, cv_input, cv_target, unsupervised_weight, logger)

      [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test, KNet_test_obs] = KNet_Pipeline.NNTest(num_test_examples, test_input, test_target, unsupervised_weight, logger)
      KNet_Pipeline.save()
      logger.plotLogger()

   #########################################################################################################################################
   #
   # # KNet with model mismatch
   # print(bcolors.HEADER + "KNet with model mismatch" + bcolors.ENDC)
   # modelFolder = 'KNet' + '/'
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   # KNet_Pipeline.setssModel(sys_model_partialh)
   # KNet_model = KalmanNetNN()
   # KNet_model.Build(sys_model_partialh)
   # KNet_Pipeline.setModel(KNet_model)
   # # KNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=10, learningRate=1e-3, weightDecay=1e-4)  # Original Settings
   # KNet_Pipeline.setTrainingParams(n_Epochs=numEpochs, n_Batch=10, learningRate=1e-3, weightDecay=1e-4,unsupervised_weight=0.5)
   #
   # # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet_obsmis_rq1030_T2000.pt",map_location=dev)
   #
   # KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test, KNet_test_obs] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   #
   #
   #
   # KNet_Pipeline.save()

   # # Save trajectories
   # # trajfolderName = 'KNet' + '/'
   # # DataResultName = traj_resultName[rindex]
   # # # EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
   # # # EKF_Partial_sample = torch.reshape(EKF_out_partial[0,:,:],[1,m,T_test])
   # # # target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
   # # # input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
   # # # KNet_sample = torch.reshape(KNet_test[0,:,:],[1,m,T_test])
   # # torch.save({
   # #             'KNet': KNet_test,
   # #             }, trajfolderName+DataResultName)

   # ## Save histogram
   # EKFfolderName = 'KNet' + '/'
   # torch.save({'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
   #             'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
   #             'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partial,
   #             'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
   #             # 'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
   #             # 'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
   #             'KNet_MSE_test_linear_arr': KNet_MSE_test_linear_arr,
   #             'KNet_MSE_test_dB_avg': KNet_MSE_test_dB_avg,
   #             }, EKFfolderName+EKFResultName)

   





