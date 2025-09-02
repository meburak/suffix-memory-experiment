import os
import pandas as pd 

folderBase = os.getcwd()
dataFolderName = "cleanData"
dataFolderPath = os.path.join(folderBase,dataFolderName)
outputData_clean = pd.DataFrame(columns=["Participant ID","Participant Group","ListID","Presented Word","Present Position","Recalled Word","Recall Position","Reaction Time","Hit"])
outputData_intrusion = pd.DataFrame(columns=["Participant ID","Participant Group","ListID","Presented Word","Present Position","Recalled Word","Recall Position","Reaction Time","Hit"])

for file in os.listdir(dataFolderPath):
    participantId = file.split("-")[0]
    participantGroup = file.split("-")[1][:3]
    inputData = pd.read_csv(os.path.join(dataFolderPath,file))

    if "clean" in file: 
        outputData_clean = pd.concat([inputData, outputData_clean], ignore_index=True)
        outputData_clean["Participant ID"].fillna(f"{participantId}",inplace=True)
        outputData_clean["Participant Group"].fillna(f"{participantGroup}",inplace=True)
    
    if "intrusion" in file: 
        outputData_intrusion = pd.concat([inputData, outputData_intrusion], ignore_index=True)
        outputData_intrusion["Participant ID"].fillna(f"{participantId}",inplace=True)
        outputData_intrusion["Participant Group"].fillna(f"{participantGroup}",inplace=True)

    outputData_clean.to_csv(os.path.join(dataFolderPath,"CombinedClean.csv"), index=False, encoding="utf-8-sig")
    outputData_intrusion.to_csv(os.path.join(dataFolderPath,"CombinedIntrusion.csv"), index=False, encoding="utf-8-sig")
        
        
