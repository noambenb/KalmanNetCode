import os
from os import path as path
import numpy as np
import matplotlib.pyplot as plt

class Logger:

    def __init__(self, strTime, folderName, modelName, unsupervised_weight):
        self.strTime = strTime.replace('.', '-')
        self.strTime = self.strTime.replace(':', '-')

        self.folderName = folderName + '/'
        self.modelName = modelName

        # create directory Logs if does not exist
        if not path.exists(self.folderName):
            os.mkdir(self.folderName)

        self.folderName = self.folderName + self.strTime[0:8] + '/'
        if not path.exists(self.folderName):
            os.mkdir(self.folderName)


        self.logFileName = modelName + "_" + "usw" + str(unsupervised_weight) + "_" + self.strTime[9:] + ".txt"
        self.logFile = open(self.folderName + self.logFileName, "w")
        self.logFile.write(modelName + "\n")
        self.logFile.close()


    def logEntry(self, stringEntry):
        self.logFile = open(self.folderName + self.logFileName, "a")
        stringEntry = stringEntry.replace('tensor(', ' ')
        stringEntry = stringEntry.replace(')', ' ')
        self.logFile.write(stringEntry+"\n")
        self.logFile.close()

    def plotFromFile(self, file_path):

        if path.isfile(file_path):
            with open(file_path, 'r') as file:
                # read the entire text file to a single string
                data = file.read() #.replace('\n', '')

                # get the Unsupervised Weight value
                search_word = "Unsupervised Weight = "
                start_ind = data.find(search_word) + len(search_word)
                search_word = "\n"
                end_ind = data.find(search_word, start_ind)
                usw = float(data[start_ind:end_ind])

                # get the total number of Epochs
                search_word = "1/"
                start_ind = data.find(search_word) + len(search_word)
                search_word = " - "
                end_ind = data.find(search_word, start_ind)
                num_epochs = int(data[start_ind:end_ind])

                mse_training_vec = np.zeros(num_epochs)
                mse_validation_vec = np.zeros(num_epochs)

                for i_epoch in range(0, num_epochs):
                    start_epoch_line = str(i_epoch+1) + "/" + str(num_epochs)
                    start_epoch_line_ind = data.find(start_epoch_line)
                    # get MSE Training for current epoch
                    search_word = "MSE Training: "
                    start_ind = data.find(search_word, start_epoch_line_ind) + len(search_word)
                    search_word = " [dB] "
                    end_ind = data.find(search_word, start_ind)
                    mse_training_vec[i_epoch] = float(data[start_ind:end_ind])

                    # get MSE Validation:   for current epoch
                    search_word = "MSE Validation: "
                    start_ind = data.find(search_word, end_ind) + len(search_word)
                    search_word = "[dB]"
                    end_ind = data.find(search_word, start_ind)
                    mse_validation_vec[i_epoch] = float(data[start_ind:end_ind])

                # get optimal point and index
                search_word = "Optimal Validation idx:"
                start_ind = data.find(search_word) + len(search_word)
                search_word = " Optimal Validation:"
                end_ind = data.find(search_word, start_ind)
                optimal_index = float(data[start_ind:end_ind])

                search_word = "Optimal Validation:  "
                start_ind = data.find(search_word) + len(search_word)
                search_word = " [dB]"
                end_ind = data.find(search_word, start_ind)
                optimal_value = float(data[start_ind:end_ind])

                plt.style.use('classic')
                fig, ax = plt.subplots()
                # plot the training vector
                ax.plot(range(1, num_epochs+1), mse_training_vec, 'b-', label='Training')
                # plot the validation vector
                ax.plot(range(1, num_epochs+1), mse_validation_vec, 'g-', label='Validation')
                # plot the minimal point
                plt.plot([optimal_index], [optimal_value], marker='o', markersize=6, color="red")

                # Plot axes labels and show the plot
                plt.xlabel('Epoch index')
                plt.ylabel('MSE [dB]')
                ax.legend(loc='upper right', frameon=False)



                plt.show()
                x = 1


        else:
            print("Error! file does not exist:")
            print(file_path)

    def plotLogger(self):
        file_path = self.folderName + self.logFileName
        self.plotFromFile(file_path)

