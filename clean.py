import os 
import pandas as pd
import Levenshtein as lv
import unicodedata
import numpy as np
from openpyxl.utils import column_index_from_string as cifs
import configparser 

columnsNumbers = []

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
    print(f"Columns converted to index numbers. \n Numbers:: {columnsNumbers}")

def dropColumns(x): #for removing the columns we did not want
    tempRemove=[]
    for i in range(len(x.columns)): 
        if i not in columnsNumbers: 
            tempRemove.append(x.columns[i])
        else:
            continue
    for i in tempRemove:
        x = x.drop(columns=[f"{i}"])
    print("Columns Dropped!")
    return x

def rowDivider(x, col:str): #x = data col=seperator column "Words"
    
    x[col] = x[col].fillna(0)

    div = {}
    tempList = []
    listNumber = 0

    for i in range(len(x[col])-1):
        if listNumber <= numLists: #there is a problem with experiments output, where it gives an extra list output. 
            p = x[col][i] #this will ensure that does not happen 
            pDiff = x[col][i+1]
            
            if p == 0 and pDiff != p: 
                #print("Start Point added")
                tempList.append(i)
    
            if p != 0 and pDiff == 0: 
                #print("End Point Added")
                tempList.append(i)
                div.update({f"List-00{listNumber}": tempList})
                #print("Dictionary Updated!")
                tempList = []
                listNumber += 1
                
            else:
                continue

    return div

def parseWords(df, col:str,): #Create list-word match
    indexes = rowDivider(df,col) #
    indexes
    wordsParsed = {}
    tempList = []
    for key in indexes.keys():
        
        tempList = []
        x, y = indexes.get(key)

        for q in range(x+1,y+1):
            word = df[col][q]
            tempList.append(word)
            
        wordsParsed.update({key: tempList})
    return wordsParsed

def findInputLists(df):
    recallDictionary = {}
    rtDictionary = {}
    rec = 0
    rt = 0
    df = df.fillna(0)

    #naArray = np.full(shape=len(df["Words"]),fill_value="NaN")

    for col in df.columns:
        if col.endswith(".text") == True:
            recallDictionary.update({f"List-00{rec}": df[col]})
            rec += 1 
        if col.endswith(".rt") == True: 
            rtDictionary.update({f"List-00{rt}": df[col]})
            rt += 1
        else: 
            continue
      

    if recallDictionary is None:
        print(f"{RED}BROKEN FILE! Missing recallDictionary.{RESET}")
        return 0, 0
        
    if rtDictionary is None:
        print(f"{RED}BROKEN FILE! Missing rtDictionary.{RESET}")
        return 0, 0
    
    else:
        return recallDictionary, rtDictionary
    
def calcSerialPos(rc, rt,file):
    #extract the position of each words presentation
    if rc == 0 or rt == 0: 
        print(f"{RED}BROKEN FILE! {file}{RESET}")
        return 0, 0
        
    
    else:
        divrc = {}
        divrt = {}
        for key in rc.keys(): #recall #now, i am implementing rt to this. 
                            #if the rt and rc had different keys, it would be problematic to do
                            #what i am about to do, but they are same so we can proceed. 
            ###Find the Indexes
            tempList = rc.get(key)
            tempList = pd.DataFrame(tempList) #so I can use rowIndexer
            #Imported rowIndexer part
            tempList = tempList.fillna(0)
            col = tempList.columns[0]
            idXList = []

            for i in range(len(tempList[col])-1): #for finding out start-end indexes 

                p = tempList[col][i]
                pDiff = tempList[col][i+1]
                
                if p == 0 and pDiff != p: 
                    #print("Start Point added")
                    idXList.append(i+1)

                if p != 0 and pDiff == 0: 
                    #print("End Point Added")
                    idXList.append(i)
                    
                else:
                    continue        #Okay, found the indexes. 

                #We have now: idXList = [start, end] we can convert to this
                #dictionary = {"A1":1} 
                for i in range(idXList[0],idXList[1]+1):
                    pos = i - (idXList[0] - 1)
                    newkey = tempList[col][i] #had to do that because it was giving outputs with \n, I couldnt figure out why. So easy solution
                    #well didnt need that, instead [-1] solved it. But why? I dont get it. 
                    
                    if rc is not None and rt is not None:
                        try: 
                            reactionT = float(rt.get(key)[i])

                        except Exception as e: 
                            raise Exception(f"Processing file: {file}, {str(e)}")
                    
                    divrc.update({f"{tempList[col][i][:-1]}": pos})
                    divrt.update({f"{tempList[col][i][:-1]}": reactionT})

        return divrc, divrt

            
def calcPresentPos(dct):

    for key in dct.keys(): 
        pass
    
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

###Iterate files
for file in folderRawCSV: #start the loop, it starts if a data is not already cleaned

    fileCleaned = file.strip(".csv") + "_clean.csv"
    filePath = os.path.join(folderDataRaw, file)

    if fileCleaned not in os.listdir(folderDataClean):
        #left here, continue from
        dataInput = pd.read_csv(filePath) #take file

        convertLetterstoNumbers(columnsExtract) #Now we can turn it to numbers. 
        #the number array is returned to a global list created at the start. 
        #that number array is fed into dropColumns
        dataInput = dropColumns(dataInput)

        rcDict, timeDict = findInputLists(dataInput) #create recall and reaction time dict
        words = parseWords(dataInput, "Words") #parse the words, create dictionary

        recallSP, reactSP = calcSerialPos(rcDict,timeDict,file) #serial position, indexing
        #reactSP is named as SP but has nothing to do with it.
        #just so it can be assigned from a single function

        if recallSP == 0 and reactSP == 0:
            continue

        recallSP = pd.DataFrame.from_dict(recallSP, orient='index', columns=['recallPos'])
        reactSP = pd.DataFrame.from_dict(reactSP, orient='index', columns=['reactionTime'])
        middleframe = pd.merge(recallSP,reactSP,left_index=True,right_index=True)
        middleframe = middleframe.rename_axis("recalledWords").reset_index()
        #created a df object for recall words

        px = {  
            "listID" : [],
            "presentedWords" : [],
            "presentPos" : [], 
        }
        px = pd.DataFrame(px)

        idX = 0
        for key in words.keys():
            n = 1 
            for item in words.get(key):
                px.loc[idX] = [key,item,n]
                n += 1 
                idX += 1        

        lengthW = len(px["presentedWords"])
        naArray = np.full(shape=lengthW,fill_value=None)
        px["recalledWords"] = naArray
        px["recallPos"] = naArray
        px["reactionTime"] = naArray
        
        lengthR = len(middleframe["recalledWords"])
        for i in range(lengthW):
            valW = px["presentedWords"][i]
            for r in range(lengthR):
                valR = middleframe["recalledWords"][r]
                if valR == valW: 
                    #px["recalledWords"][i] = middleframe["recalledWords"][r]
                    #px["recallPos"][i] = middleframe["recallPos"][r]
                    #px["reactionTime"][i] = middleframe["reactionTime"][r]

                    px.loc[i,"recalledWords"] = middleframe["recalledWords"][r]
                    px.loc[i,"recallPos"] = middleframe["recallPos"][r]
                    px.loc[i,"reactionTime"] = middleframe["reactionTime"][r]


                else:
                    continue

        px

        #save the files

        
        output_path = os.path.join(folderDataClean,fileCleaned)
        px.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"{GREEN}File created! {file}{RESET}")






    if fileCleaned in os.listdir(folderDataClean):
        print(f"{BLUE}File with name: {fileCleaned}, already exists in cleanData as {file}_clean.csv{RESET}")

        continue

