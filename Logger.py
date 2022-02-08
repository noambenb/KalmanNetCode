import os
from os import path as path

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





