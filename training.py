#!/usr/bin/env python
# coding: utf-8

# In[5]:


import importlib
import preprocessing
import torch
importlib.reload(preprocessing)

# DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
DEVICE=torch.device('cpu')
loaders = preprocessing.TrainLoader(DEVICE)
trainloader = loaders.trainloader
valloader = loaders.valloader
dic = preprocessing.dic


# In[6]:


import encoder
import decoder


# In[34]:


import importlib
importlib.reload(decoder)
importlib.reload(encoder)
import os
import pickle
import time
import numpy as np


import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

EncoderCNN = encoder.EncoderCNN
SentenceRNN = decoder.SentenceRNN
WordRNN = decoder.WordRNN

EPOCHS = 2
LEARNING_RATE = 0.1
BATCH_SIZE=2

class Im2pGenerator(object):
    def __init__(self):
        print('in constructor')
        self.train_data_loader = trainloader
        self.val_data_loader = trainloader
        self.encoderCNN = EncoderCNN()
        self.sentenceRNN = SentenceRNN()
        self.wordRNN = WordRNN(hidden_size=256, vocab_size=len(dic), att_dim=256, embed_size=256, encoded_dim=256, device=DEVICE)
        self.criterionSentence = nn.BCELoss(size_average=False, reduce=False)
        self.criterionWord = nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(params=(
            list(self.encoderCNN.parameters()) + list(self.sentenceRNN.parameters()) + list(self.wordRNN.parameters())
        ), lr=LEARNING_RATE)

    def train(self):
        for epoch in range(EPOCHS):
            train_loss = self.__epoch_train()
#             val_loss = self.__epoch_val()
            val_loss = 0
            print("[{}] Epoch-{} - train loss:{} - val loss:{} - lr:{}".format(self.__get_now(),
                                                                               epoch + 1,
                                                                               train_loss,
                                                                               val_loss,
#                                                                                self.optimizer.param_groups[0]['lr']
                                                                               0
                                                                              ))
            self.__save_model(epoch)

    def __epoch_train(self):
        print('in epoch train')
        train_loss = 0
        self.encoderCNN.train()
        self.wordRNN.train()
        self.sentenceRNN.train()
        self.encoderCNN.to(DEVICE)
        self.wordRNN.to(DEVICE)
        self.sentenceRNN.to(DEVICE)
        
        for i, (images, findings, sentenceVectors, word2d, wordsLengths) in enumerate(self.train_data_loader):
            images = images.to(DEVICE)
            word2d = word2d.to(DEVICE)
            featureMap, globalFeatures = self.encoderCNN.forward(images)
            sentence_states = None
            loss = 0
            sentenceLoss = 0
            wordLoss = 0
            word2d = word2d.permute(1, 0, 2) #(sentenceIndex, batchSize, maxWordsInSentence)
            for sentenceIndex, sentence_value in enumerate(sentenceVectors):
                endToken, topic_vec, sentence_states = self.sentenceRNN.forward(globalFeatures, sentence_states)
                endToken = endToken.squeeze(1).squeeze(1)

                """***TODO*** Should stop calculating loss for sentences once they're done."""
                sentenceLoss = sentenceLoss + self.criterionSentence(endToken, sentence_value.type(torch.float).to(DEVICE)).sum()
                
                captions=word2d[sentenceIndex]
                captionLengths=wordsLengths[sentenceIndex]
                if(any(captionLengths)):
                    predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = self.wordRNN.forward(
                        enc_image=featureMap,
                        global_features=globalFeatures,
                        encoded_captions=captions,
                        caption_lengths=captionLengths
                    )
                    # predictions: (batch_size, largest_sentence_in_batch_size, vocab_size)
                    targets = captions

                    # Remove timesteps that we didn't decode at, or are pads
                    # pack_padded_sequence is an easy trick to do this

                    greaterThan0LengthIndeces = list() #remove length 0 sentences
                    greaterThan0Lengths = list()
                    for i, length in enumerate(decode_lengths):
                        if(length > 0):
                            greaterThan0LengthIndeces.append(i)
                            greaterThan0Lengths.append(length)
                    
                    targets = targets[greaterThan0LengthIndeces]
                    predictions = predictions[greaterThan0LengthIndeces]
                    targets = pack_padded_sequence(targets, greaterThan0Lengths, batch_first=True).data

                    scores = pack_padded_sequence(predictions, greaterThan0Lengths, batch_first=True).data
                    
                    
                    # Calculate loss
                    wordLoss = wordLoss + self.criterionWord(scores, targets)
                    
            loss = wordLoss + sentenceLoss
            self.optimizer.zero_grad()
            # Update weights
            loss.backward()
            self.optimizer.step()
            break

        return train_loss
    
    def __epoch_val(self):
        print('in epoch val')
        val_loss = 0
        for i, (images, findings, sentenceVectors, word2d, wordsLengths) in enumerate(self.train_data_loader):
            images = images.to(DEVICE)
            word2d = word2d.to(DEVICE)
            featureMap, globalFeatures = self.encoderCNN.forward(images)
            sentence_states = None
            loss = 0
            sentenceLoss = 0
            wordLoss = 0
            
            for sentenceIndex, sentence_value in enumerate(sentenceVectors):
                endToken, topic_vec, sentence_states = self.sentenceRNN.forward(globalFeatures, sentence_states)
                endToken = endToken.squeeze(1).squeeze(1)
              
                captions=word2d[sentenceIndex]
                captionLengths=wordsLengths[sentenceIndex]
                if(any(captionLengths)):
                    predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = self.wordRNN.forward(
                        enc_image=featureMap,
                        global_features=globalFeatures,
                        encoded_captions=captions,
                        caption_lengths=captionLengths
                    )
                    # predictions: (batch_size, largest_sentence_in_batch_size, vocab_size)
                    targets = captions

                    # Remove timesteps that we didn't decode at, or are pads
                    # pack_padded_sequence is an easy trick to do this

                    greaterThan0LengthIndeces = list() #remove length 0 sentences
                    greaterThan0Lengths = list()
                    for i, length in enumerate(decode_lengths):
                        if(length > 0):
                            greaterThan0LengthIndeces.append(i)
                            greaterThan0Lengths.append(length)
                    targets = targets[greaterThan0LengthIndeces]
                    predictions = predictions[greaterThan0LengthIndeces]
                    
                    # Need to softmax predictions to generate vocabular index
                    # Back convert targets and predictions to the proper sentence
                    # Add to references and hypotheses
                    
            break
        # compare references and hypotheses to generate bleu
        
        return val_loss

    def __get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def __get_now(self):
        return str(time.strftime('%y%m%d-%H:%M:%S', time.gmtime()))
    
    def __save_model(self, epoch):
        state = {'epoch': epoch,
                 'sentenceRNN': self.sentenceRNN,
                 'wordRNN': self.wordRNN,
                 'encoderCNN': self.encoderCNN,
                 'optimizer': self.optimizer}
        filename = 'checkpoints/checkpoint_' + str(epoch) + '.pth.tar'
        torch.save(state, filename)

im2p = Im2pGenerator()
im2p.train()

