import os
import torch
import torch.nn as nn
import random
import Logger
from Plot import Plot
from datetime import datetime
import matplotlib.pyplot as plt


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay, unsupervised_weight=0):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay
        self.unsupervised_weight = unsupervised_weight

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    # cv_input - [y_t]_1_^T
    def NNTrain(self, n_Examples, n_Labeled_Examples, train_input, train_target, n_CV, cv_input, cv_target, unsupervised_weight, logger):
        """
        train_input     - are the samples of the target, denoted by: y_t.
        train_target    - are the state vector of the target, denoted by: x_t.
        cv_input        - are the same as train_input, but used for the validation phase.
        cv_target       - are the same as train_target, but used for the validation phase.
        """

        str1 = "Num Labeled Examples in Training: " + str(n_Labeled_Examples)
        print(str1)
        logger.logEntry(str1)
        str1 = "Num Examples in Cross Validation: " + str(n_CV)
        print(str1)
        logger.logEntry(str1)

        self.N_E = n_Examples
        self.N_CV = n_CV

        # initialize empty areas
        MSE_cv_linear_batch = torch.empty([self.N_CV])
        MSE_cv_linear_batch_obs = torch.empty([self.N_CV])

        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        # self.MSE_cv_linear_epoch_obs = torch.empty([self.N_Epochs])

        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])
        # self.MSE_cv_dB_epoch_obs = torch.empty([self.N_Epochs])
        # self.MSE_cv_dB_epoch_hybrid_loss = torch.empty([self.N_Epochs])

        MSE_train_linear_batch = torch.empty([self.N_B])
        MSE_train_linear_batch_obs = torch.empty([self.N_B])

        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        num_supervised_iterations = 0

        for ti in range(0, self.N_Epochs):

            #################################
            ### Validation Sequence Batch ###
            #################################

            start_time = datetime.now()

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T_test)

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T_test)
                y_out_cv = torch.empty(self.ssModel.n, self.ssModel.T_test)

                for t in range(0, self.ssModel.T_test):
                    x_out_cv[:, t] = self.model(y_cv[:, t])
                    y_out_cv[:, t] = self.model.m1y.squeeze().T

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item()
                # MSE_cv_linear_batch_obs[j] = self.loss_fn(y_out_cv, y_cv[:, :]).item()

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            # self.MSE_cv_linear_epoch_obs[ti] = torch.mean(MSE_cv_linear_batch_obs)
            # self.MSE_cv_dB_epoch_obs[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch_obs[ti])

            # Hybrid Loss
            # self.MSE_cv_dB_epoch_hybrid_loss[ti] = self.MSE_cv_dB_epoch[ti] * (1 - unsupervised_weight) + \
            #               self.MSE_cv_dB_epoch_obs[ti] * unsupervised_weight

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0


            supervised_input = train_input[:n_Labeled_Examples, :, :]
            unsupervised_input = train_input[n_Labeled_Examples:, :, :]
            supervised_prob = n_Labeled_Examples / n_Examples   # should be according to the size of the labeled examples from all examples?

            for j in range(0, self.N_B):
                supervised_iteration = random.random() < supervised_prob
                if supervised_iteration:
                    num_supervised_iterations = num_supervised_iterations + 1
                    n_e = random.randint(0, supervised_input.shape[0]-1)
                    y_training = supervised_input[n_e, :, :]
                else:
                    n_e = random.randint(0, unsupervised_input.shape[0]-1)
                    y_training = unsupervised_input[n_e, :, :]
                # n_e = random.randint(0, self.N_E - 1)
                # y_training = train_input[n_e, :, :]
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                y_out_training = torch.empty(self.ssModel.n, self.ssModel.T)

                for t in range(0, self.ssModel.T):
                    x_out_training[:, t] = self.model(y_training[:, t])
                    y_out_training[:, t] = self.model.m1y.squeeze().T

                # Compute Training Loss
                if supervised_iteration:
                    LOSS = self.loss_fn(x_out_training, train_input[n_e, :, :])
                else:
                    LOSS = self.loss_fn(y_out_training, train_target[n_e, :, :])

                # LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :])
                # LOSS_obs = self.loss_fn(y_out_training, y_training[:, :])
                # Loss_hybrid = LOSS * (1 - unsupervised_weight) + LOSS_obs * unsupervised_weight

                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            logger.logEntry(str(ti + 1) + "/" + str(self.N_Epochs) + " - MSE Training:" + str(self.MSE_train_dB_epoch[ti]) + "[dB] " + "MSE Validation: " + str(self.MSE_cv_dB_epoch[ti]) + "[dB]")
            print(bcolors.BOLD + bcolors.UNDERLINE + str(ti + 1), "- MSE Training:" + bcolors.ENDC, self.MSE_train_dB_epoch[ti], "[dB]",
                        "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]")

            end_time = datetime.now()
            elapsed_time = end_time - start_time
            print("Elapsed time for Epoch ", ti + 1, "/", self.N_Epochs, " is: ", elapsed_time)
            print("Number of Supervised Iterations is: " + str(num_supervised_iterations) + " out of: " + str(ti + 1))
            if torch.isnan(self.MSE_train_dB_epoch[ti]) or torch.isnan(self.MSE_cv_dB_epoch[ti]):
                print(bcolors.FAIL + "NaN results! exiting..." + bcolors.ENDC)
                logger.logEntry("NaN results! exiting...")
                logger.plotLogger()
                # os.system('ruAgain.bat')
                print(quit)
                quit()

            if (ti > 0):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                if (d_train < 0) and (d_cv < 0):
                    print(bcolors.OKCYAN + "diff MSE Training :", d_train, "[dB]",
                          "diff MSE Validation :", d_cv, "[dB]" + bcolors.ENDC)
                else:
                    print(bcolors.WARNING + "diff MSE Training :", d_train, "[dB]",
                          "diff MSE Validation :", d_cv, "[dB]" + bcolors.ENDC)

            print(bcolors.UNDERLINE + "Optimal Validation idx:", self.MSE_cv_idx_opt + 1, "Optimal Validation:", self.MSE_cv_dB_opt, "[dB]" + bcolors.ENDC)

        logger.logEntry("Optimal Validation idx:" + str(self.MSE_cv_idx_opt + 1) + " Optimal Validation: " + str(self.MSE_cv_dB_opt) + "[dB]")
        logger.logEntry("Number of supervised iterations:" + str(num_supervised_iterations) + "/" + str(self.N_E) + "[dB]")

        logger.logEntry2("Num Examples Training:" + str(self.N_E) + " Optimal Validation: " + str(self.MSE_cv_dB_opt) + "[dB]")
        logger.set_num_supervised_itterations(num_supervised_iterations)


    def NNTest(self, n_Test, test_input, test_target, unsupervised_weight, logger):

        self.N_T = n_Test

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        self.model = torch.load(self.modelFileName)

        self.model.eval()

        torch.no_grad()

        x_out_array = torch.empty(self.N_T,self.ssModel.m, self.ssModel.T_test)
        y_out_array = torch.empty(self.N_T,self.ssModel.n, self.ssModel.T_test)

        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j, :, :]

            self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T_test)

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T_test)
            y_out_test = torch.empty(self.ssModel.n, self.ssModel.T_test)

            for t in range(0, self.ssModel.T_test):
                x_out_test[:, t] = self.model(y_mdl_tst[:, t])
                y_out_test[:, t] = self.model.m1y.T

            loss_supervised = loss_fn(x_out_test, test_target[j, :, :])
            # loss_unsupervised = loss_fn(y_out_test, test_input[j, :, :])
            # loss_hybrid = loss_supervised * (1 - unsupervised_weight) + \
            #               loss_unsupervised * unsupervised_weight

            self.MSE_test_linear_arr[j] = loss_supervised.item()

            x_out_array[j, :, :] = x_out_test
            y_out_array[j, :, :] = y_out_test

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Print MSE Cross Validation
        str1 = self.modelName + "-" + "MSE Test:"
        print(str1, self.MSE_test_dB_avg, "[dB]")
        logger.logEntry(str1 + str(self.MSE_test_dB_avg) + "[dB]")
        logger.logEntry2(str1 + str(self.MSE_test_dB_avg) + "[dB]")


        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_array, y_out_array]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)