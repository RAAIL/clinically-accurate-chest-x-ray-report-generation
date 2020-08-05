#!/usr/bin/env python
# coding: utf-8

# In[5]:


### Data Preparation
import re
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import matplotlib.image as io 
import xmltodict
from PIL import Image

MAX_NUMBER_OF_SENTENCES=15
MAX_SENTENCE_LENGTH=55

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Doc():
    def __init__(self, fd):
        self.parsed = xmltodict.parse(fd.read())
    
    def getId(self):
        return self.parsed['eCitation']['uId']['@id']
    
    def getFindings(self):
        parsedText = self.getReportText()                    
        findings = parsedText["FINDINGS"] if "FINDINGS" in parsedText else ""
        return findings

    def getReportText(self):
        aT = self.parsed['eCitation']["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        dicAbstractText = {}
        for val in aT:
            if "#text" in val:
                dicAbstractText[val['@Label']] = val['#text']
        return dicAbstractText

def cleanWord(str):
    return re.sub("\.$|\,$", "", str).lower()

class Dictionary():
    EOS = '<eos>'
    SOS = '<sos>'
    def __init__(self):
        self.wordId = {}
        self.idWord = list()
        self.nextId = 0
        self.addWord(Dictionary.EOS)
        self.addWord(Dictionary.SOS)

    def addWord(self, word):
        cw = cleanWord(word)
        if cw not in self.wordId:
            self.wordId[cw]=self.nextId
            self.idWord.append(cw)
            self.nextId = self.nextId + 1
        return cw, self.wordId[cw]
    
    def getWordId(self, word):
        cw , _id = self.addWord(word)
        return _id
    
    def __len__(self):
        return self.nextId

    
class ImagePaths():
    def filterImages(self, imageName):
            return imageName[-3:] == "png"
        
    def __init__(self, image_dir):
        self.paths = list(filter(self.filterImages, os.listdir(image_dir)))

class Sentences():
    def __init__(self, findings):
        self.sentences = re.split('\. ', findings or '')
    
    def sentencesVector(self):
        vector =  [0]*MAX_NUMBER_OF_SENTENCES
        for i in range(len(self)):
            vector[i] = 1
        return vector
    
    def __len__(self):
        return len(self.sentences)
    
    def initWordVector(self):
        return [[dic.getWordId(Dictionary.EOS)]*MAX_SENTENCE_LENGTH for i in range(MAX_NUMBER_OF_SENTENCES)]
    
    def wordVector(self):
        error = False
        wordVector = self.initWordVector()
        wordLengths = [0]*MAX_NUMBER_OF_SENTENCES
        for i, sentence in enumerate(self.sentences):
            words = sentence.split(' ')            
            if(len(words) == 0 or len(words) > MAX_SENTENCE_LENGTH):
                error = True
            for j in range(len(words)):
                wordVector[i][j] = dic.getWordId(words[j])
            wordLengths[i] = len(words)
        return wordVector, wordLengths, error
    
    def initWordLengthsVector(self):
        wordVectorLengths = [0]*MAX_NUMBER_OF_SENTENCES
        
        
class ImagesReports(Dataset):
    def getEncodingsByReport(self):
        """
        Read through all of the reports
        """
        for i in range(len(self.reports_path)):
            with open(os.path.join(self.reports_dir, self.reports_path[i])) as fd:
                doc = Doc(fd)
                findings = doc.getFindings()
                if(findings == ""):
                    self.skipImages.add(doc.getId())
                    continue
                sentences = Sentences(findings)
                """
                if a report doesnt have any sentences, or if it has more than MAX_NUMBER_OF_SENTENCES, skip it
                otherwise build sentence vector which should have the format of [1,1,1,1,0,....] len(vector) = MAX_NUMBER_OF_SENTENCES       
                """

                if(len(sentences) < MAX_NUMBER_OF_SENTENCES and len(sentences) > 0):
                    sentenceVector = sentences.sentencesVector()
                else:
                    self.skipImages.add(doc.getId()) #skip images where the sentence is 0 or greater than max number of setences
                    continue
      
                wordVector, wordLengths, error = sentences.wordVector()
                if(error):
                    self.skipImages.add(doc.getId()) #skip images where the sentence is 0 or greater than max sentence length
                    continue
        
                self.reports[doc.getId()] = {
                    "findings": findings,
                    "encodedWordsBySentence": wordVector,
                    "encodedSentence": sentenceVector,
                    "wordsLengths": wordLengths
                }
                if(self.saveFindings):
                    self.allFindings.append(findings)

    def cleanImagePaths(self):
        tempImagesPath = []
        for imageName in self.images_path:
            reportId = imageName.split('_')[0]
            if(not (reportId in self.skipImages)):
                tempImagesPath.append(imageName)
        self.images_path = tempImagesPath
        
    def __init__(self, image_dir, report_dir, device, transform=None, saveFindings=False):
        self.device=device
        self.transform = transform
        self.images_dir = image_dir
        self.reports_dir = report_dir
        self.images_path = ImagePaths(image_dir).paths
        self.reports_path = os.listdir(report_dir)
        self.saveFindings = saveFindings
        self.allFindings = list();
        self.reports = {}
        self.skipImages = set()
        self.getEncodingsByReport()
        self.cleanImagePaths()
        if(saveFindings):
            with open('myCorpus.txt', 'w+') as f:
                for item in self.allFindings:
                    f.write("%s\n" % item)

    def __len__(self):
        return len(self.images_path)

    def getReportIdFromImagePath(self, imagePath):
        return imagePath.split('_')[0]
    
    def getImage(self, imagePath):
        fullImgPath = os.path.join(self.images_dir, imagePath)
        image = Image.open(fullImgPath)
        if self.transform:
            image = self.transform(image)
        return image
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.images_path[idx]
        image = self.getImage(imagePath)        
        reportId = self.getReportIdFromImagePath(imagePath)
        return (
            image,
            self.reports[reportId]["findings"], #raw_str
            self.reports[reportId]["encodedSentence"], #[1,1,0...] .shape = (MAX_NUMBER_OF_SENTENCES)
            torch.tensor(self.reports[reportId]["encodedWordsBySentence"]), #[[1,2,6,8, ...]] (MAX_NUMBER_OF_SENTENCES, MAX_NUMBER_OF_WORDS_IN_SENTENCES)
            self.reports[reportId]["wordsLengths"] # list(int)
        )

class TrainLoader: #Change to DataLoader because it loads everything
    def __init__(self, device, batchSize=2, saveFindings=False): # Change to one root directory #Change to crawl based approach
        dataset = ImagesReports('./data/nlm/images', './data/nlm/reports', device=device, transform=preprocess, saveFindings=saveFindings)
        trainSize = int(dataset.__len__()*0.7)
        trainSet, testAndValSet = torch.utils.data.random_split(dataset, [trainSize, dataset.__len__() - trainSize])
        valSize = int(dataset.__len__()*0.2)
        valSet, testSet = torch.utils.data.random_split(testAndValSet, [valSize, dataset.__len__() - trainSize - valSize])
        self.trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize,
                                          shuffle=True, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(valSet, batch_size=batchSize,
                                          shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testSet, batch_size=batchSize,
                                          shuffle=True, num_workers=2)

dic = Dictionary()

