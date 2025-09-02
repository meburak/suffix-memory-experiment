import os 
import pandas as pd
import Levenshtein as lv
import unicodedata
import numpy as np
from openpyxl.utils import column_index_from_string as cifs
import configparser 
import logging
import time

columnsNumbers = []
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{time.strftime("%d_%H_%M", time.localtime())}_demo.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = 10
)

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


def normalize(text): #normalize to unicode, latin letters
    return (
        unicodedata.normalize('NFKD', text.upper())
        .replace('İ', 'I')
        .replace('Ş', 'S')
        .replace('Ğ', 'G')
        .replace('Ü', 'U')
        .replace('Ö', 'O') #İYİLİK - IYILIG 
        .replace('Ç', 'C')
        .encode('ASCII', 'ignore')
        .decode('utf-8')
    )
def listCSV(lst):
    listOut = []
    for file in lst: 
        if file.endswith(".csv"):
            listOut.append(file)
        else:
            continue
    return listOut

def levenshtein(word, target):
    word = normalize(word)
    target = normalize(target) #Normalize input
    dist = lv.distance(word, target)
    if dist <= numTolerate: 
        return True
    

def convertLetterstoNumbers(lst): #For converting the letter inputs into values we can iterate through,
    #and that will match with the indexing of the data
    columnsNumbers.clear()
    for str in lst: 
        idX = cifs(str)-1
        columnsNumbers.append(idX)
    logger.info(f"Columns converted to index numbers. \n Numbers:: {columnsNumbers}")

def dropColumns(x): #for removing the columns we did not want
    tempRemove=[]
    for i in range(len(x.columns)): 
        if i not in columnsNumbers: 
            tempRemove.append(x.columns[i])
        else:
            continue
    for i in tempRemove:
        x = x.drop(columns=[f"{i}"])
    logger.info("Columns Dropped!")
    return x

def columnStrip(df,col,char):
    length = len(df[col])
    for i in range(length): 
        df.loc[i, col] =  df.loc[i, col].strip(char)
    return df

def noneFound():
    pass

def findColumnTypes(df, searchedType):
    temp = [] #returns names of columns with desired ending
    for column in df.columns:
        if column.endswith(searchedType): 
            temp.append(column)
    
    if len(temp) <4: #rt are not  
        logger.error(f"{temp}, unsatisfactory column count.")
        return temp
    logger.info(f"Columns found for: {searchedType}, \n and are {temp}")
    return temp

def isOneCellMP(cell): #checks if a single cell is one cell multiple response type

    if type(cell) == str:
            splitList = cell.split(" ")
            if len(splitList) > 1 and len(splitList[1]) > 1:
                return True
            else:
                return False
    
def arrayCombine(array): #combines the inputs in the array 
    rows = [] #to be used for extracting rt values
    combined = []
    for i in range(len(array)):
        cell = array[i]
        if isOneCellMP(cell) == True:
            dct = cell.split(" ")
            for word in dct: 
                combined.append(word)
                rows.append(i)
        #logger.info("Cell Sepeerated")

        if isOneCellMP(cell) == False and type(cell) == str: 
            combined.append(cell)
            rows.append(i)
    #logger.info(f"Column seperated and combined: {combined, rows} ")

    return combined, rows
# def indexOneCellMultipleResponse(array):
#     tmp = array
#     output = []
#     for i in range(len(tmp)): 
#         cell = tmp[i]   
#         if isOneCellMP(cell) == True:
#             output.append(i)
#         else: continue
        
#     return output

# def OneCellDeconstruct(array): 
#     newArray = []
#     for cellnum in indexOneCellMultipleResponse(array): 
#         cellSplit = array[cellnum].split(" ")
#         for word in cellSplit: 
#             newArray.append(word)
#     return newArray

def createRecallRtimeTable():
    RecallRtimeTable = pd.DataFrame(columns=["ListID","Recalled Word","Recall Position","Reaction Time","RT Viable"])
    return RecallRtimeTable

def fillRecallRtimeTable(df,table,RecallEnd,ReactionEnd): 
    temp = table
    recallColumnNames = findColumnTypes(df, RecallEnd)
    pushtoRow = 0 #which row to push to
    for ncol in range(len(recallColumnNames)): #iterate through the recall cols list
        recallCol = recallColumnNames[ncol] #get the name, ncol is  used to have the listID
        array = df[recallCol] #set the array 
        recallWords, rows = arrayCombine(array) #get combined array. with row ids to use in getting reaction times.
        for i in range(len(recallWords)): 
            temp.loc[pushtoRow, "ListID"] = ncol
            temp.loc[pushtoRow, "Recalled Word"] = recallWords[i]
            temp.loc[pushtoRow, "Recall Position"] = i+1
            temp.loc[pushtoRow, "Reaction Time"] = rows[i]
            logger.info(f"To {pushtoRow}:: {ncol,recallWords[i],i+1,rows[i]} pushed.")
            pushtoRow += 1
            

    
    reactionColumnNames = findColumnTypes(df, ReactionEnd) #take the columns with endind 
    #gives an error when none found, we must take it in mind and create some instance for it.

    for i in range(len(temp["Recalled Word"])):
        gettable = temp["ListID"][i] #for the given row, take the table id 
        getindex = temp["Reaction Time"][i] #for the given row, take the row number from reaction time column. where the previous function wrote. 
        ##ERROR 
        #HANDLE FOR NO REACTION COLUMNS
        if len(reactionColumnNames) <5:
            reactionCol = [0 for _ in range(500)] #create this so we cann fill and dont have error. 
        else:
            reactionCol = df[reactionColumnNames[gettable]] #take the relevant reaction time column
        temp.loc[i,"Reaction Time"] = reactionCol[getindex] #take the value 

    temp = columnStrip(temp, "Recalled Word", "\n")
    
    return temp

def whichRowstoLists(df, columnName: str, listNum: int):
    """
    Returns the row indexes where a list starts and ends. 
    [start, end, start, end...]

    Args: 
    columnName: string type name, where the function will work
    listNum: Number of lists to look for, there is an overflow problem of one extra list. Used for that. Exclude trial list if you have one on this count. 
    
    """
    currentListCount = 0
    rowNumbersList = []
    length = len(df[columnName])
    array = df[columnName]

    for i in range(length-1):
        Left = array[i]
        Right = array[i+1]
        if pd.isna(Left) == True and pd.isna(Right) == False:
            if currentListCount <= listNum: 
                rowNumbersList.append(i+1)
            else: 
                continue
        if pd.isna(Left) == False and pd.isna(Right) == True: 
            if currentListCount <= listNum: 
                rowNumbersList.append(i)
                currentListCount += 1
            else: 
                continue
    return rowNumbersList

def createWordPresentTable():
    return pd.DataFrame(columns=["ListID","Words Presented","Present Position"])

def fillWordPresentTable(df, columnName:str, listNum: int, table):

    rowNums = whichRowstoLists(df, columnName, listNum)
    array = df[columnName]

    currentListIndexer = 0 #to generate indexes from rowNums lit
    currentListIdentifier = 0 #to write at column
    builderRowNum = 0 #for using at .loc

    for x in range(listNum+1):
        stratRowNum = int(rowNums[currentListIndexer])
        endRowNum = int(rowNums[currentListIndexer+1])

        for i in range(stratRowNum,endRowNum+1):
            table.loc[builderRowNum, "ListID"] = currentListIdentifier
            table.loc[builderRowNum, "Words Presented"] = array[i]
            table.loc[builderRowNum, "Present Position"] = (i - stratRowNum) + 1
            builderRowNum += 1
        
        currentListIndexer += 2
        currentListIdentifier += 1
    return table 

def mergeTables(wordPresent, recall):
    newTable = pd.DataFrame(columns=["ListID","Presented Word","Present Position","Recalled Word","Recall Position","Reaction Time","Hit"])
    # remainderRecall = recall
    currentRow = 0 
    wordsCol = wordPresent["Words Presented"]
    recallCol = recall["Recalled Word"]

    for i in range(len(wordsCol)): 
        presented = normalize(str(wordsCol[i]))
        presentedListRelation = wordPresent["ListID"][i]

        newTable.loc[currentRow, "ListID"] = wordPresent["ListID"][i]
        newTable.loc[currentRow, "Presented Word"] = wordPresent["Words Presented"][i]
        newTable.loc[currentRow, "Present Position"] = wordPresent["Present Position"][i]
        
        found = False

        toDrop = []

        for x in range(len(recallCol)): 
            recalled = normalize(str(recallCol[x]))
            recallListRelation = recall["ListID"][x]

            if recalled == presented and recallListRelation == presentedListRelation and found == False:
                newTable.loc[currentRow, "Recalled Word"] = recall["Recalled Word"][x]
                newTable.loc[currentRow, "Recall Position"] = recall["Recall Position"][x]
                newTable.loc[currentRow, "Reaction Time"] = recall["Reaction Time"][x]
                newTable.loc[currentRow, "Hit"] = 1 # true hit
                toDrop.append(x)
                # try:
                #     remainderRecall = remainderRecall.drop(index = toDrop).reset_index(drop=True) #to take in remaining values
                # except Exception as e:
                #     logger.error(f"While on File: {file}, \n and on function merge \n index = x and array = {remainderRecall}\n faced with Exception: {e}")

                currentRow += 1 
                found = True
                

            if recalled == presented and recallListRelation != presentedListRelation and found == False:
                newTable.loc[currentRow, "Recalled Word"] = recall["Recalled Word"][x]
                newTable.loc[currentRow, "Recall Position"] = recall["Recall Position"][x]
                newTable.loc[currentRow, "Reaction Time"] = recall["Reaction Time"][x]
                newTable.loc[currentRow, "Hit"] = 2 #list intrusion
                # remainderRecall = remainderRecall.drop(index = x) #to take in remaining values
                currentRow += 1
                found = True
            
        if found == False: 
            newTable.loc[currentRow, "Hit"] = 0 #list intrusion
            currentRow += 1
    
    return newTable

###Define folders
folderBase = os.getcwd() #/../psychopy-recall-suffix
folderDataRaw = os.path.join(folderBase, "data")
folderRawCSV = listCSV(os.listdir(folderDataRaw))

#have config
pathConfig = os.path.join(folderBase,"config.ini")
config = configparser.ConfigParser()
config.read(pathConfig)

numTolerate = int(config["DEFAULT"]["numTolerate"])
numLists = int(config["DEFAULT"]["numLists"])  # Need to match key name
columnsExtract = config["DEFAULT"].get("columnsExtract").replace(" ", "").split(",")
trialyes = int(config["DEFAULT"]["trialyes"])

if "cleanData" not in os.listdir(folderBase): #create the cleanData folder
    os.mkdir("cleanData")

folderDataClean = os.path.join(folderBase, "cleanData") 

for file in folderRawCSV: #start the loop, it starts if a data is not already cleaned

    fileCleaned = file.replace(".csv","") + "_clean.csv"
    fileRemainder = file.replace(".csv","") + "_intrusions.csv"
    filePath = os.path.join(folderDataRaw, file)

    if fileCleaned not in os.listdir(folderDataClean):
        #left here, continue from
        try:
            dataInput = pd.read_csv(filePath) #take file

            convertLetterstoNumbers(columnsExtract) #Now we can turn it to numbers. 
            #the number array is returned to a global list created at the start. 
            #that number array is fed into dropColumns
            dataInput = dropColumns(dataInput)

            recallTable = fillRecallRtimeTable(dataInput, createRecallRtimeTable(),".text",".rt")
            wordsTable = fillWordPresentTable(dataInput, "Words", 4, createWordPresentTable())

            table = mergeTables(wordsTable,recallTable)

            #save the files
            
            output_path = os.path.join(folderDataClean,fileCleaned)

            table.to_csv(output_path, index=False, encoding="utf-8-sig")
            # remainder.to_csv(os.path.join(folderDataClean,fileRemainder))

            print(f"{GREEN}File created!{fileCleaned}{RESET}")
        except Exception as e: 
            print(f"{RED}Faulty file: {file}, passed.{RESET}")
            logger.error(f"RAISED EXCEPTION. {e} for file: {file}")
            continue


    if fileCleaned in os.listdir(folderDataClean):
        print(f"{BLUE}File with name: {fileCleaned}, already exists in cleanData as {file}_clean.csv{RESET}")

        continue

