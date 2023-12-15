import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import ast

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'happy':0, 'sad':1, 'neutral':2, 'angry':3, 'excioted':4, 'frustrated':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, classify, train=True):
        self.videoIDs, self.videoSpeakers, self.emotion_labels, self.videoText,\
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid, self.sentiment_labels = pickle.load(open(path, 'rb'))
        
        if classify == 'emotion':
            self.videoLabels = self.emotion_labels
        else:
            self.videoLabels = self.sentiment_labels
        '''
        emotion_label_mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        setiment_label_mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]
        

class DailyDialogueDataset(Dataset):

    def __init__(self, path, split):
        
        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return torch.LongTensor(self.InputSequence[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.ActLabels[conv])), \
                torch.LongTensor(self.ActLabels[conv]), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                self.InputMaxSequenceLength[conv], \
                conv

    def __len__(self):
        return self.len


class LLMDataset(Dataset):
    # TODO
    def __init__(self, path):
        df = pd.read_csv(path)
        # save
        self.df = df

        #self.Speakers: list of list of 1 and 0
        #self.InputSequence: list of list of array of int
        #self.InputMaxSequenceLength: list of int
        #self.ActLabels: list of list of int
        #self.EmotionLabels: list of list of int

        # columns = ['dialog', 'emotion1', 'emotion2', 'persona', 'emotion1_gpt', 
        #            'emotion2_gpt', 'emotion_by_sentence_gpt', 'train_gpt', 'dialog_parsed',
        #            'agents', 'annotations_emo1', 'annotations_emo2']
        # self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        # self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        self.len = len(df)

        with open('dailydialog/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)  

        # tokenize inputsequence
        # add inputsequence column
        self.df['dialog_parsed'] = df['dialog_parsed'].apply(ast.literal_eval)
        self.df['InputSequence'] = ''
        for i in range(len(self.df)):
            # read in parsed dialog as list of string
            parsed_dialog = self.df['dialog_parsed'][i]
            sequence = []
            for dialog in parsed_dialog:
                # tokenize dialog
                tokenized_dialog = self.tokenizer.texts_to_sequences(dialog)
                # append to InputSequence
                sequence.append(tokenized_dialog)
        self.Speakers = self.df['agents'].tolist()
        self.InputSequence = self.df['InputSequence'].tolist()
        
            

    def __getitem__(self, index):
        conv = self.keys[index]

        
        return torch.LongTensor(self.InputSequence[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.ActLabels[conv])), \
                torch.LongTensor(self.ActLabels[conv]), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                self.InputMaxSequenceLength[conv], \
                conv

    def __len__(self):
        return self.len


class LLMPadCollate:
    # TODO
    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i==0 else \
                pad_sequence(dat[i]) if i == 1 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]



class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i==0 else \
                pad_sequence(dat[i]) if i == 1 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]
