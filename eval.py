import numpy as np
from pathlib import Path
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from keras import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from model import E2ELSTMModel, MaskedNLLLoss
from dataloader import LLMPadCollate, LLMDataset, DailyDialogueDataset, DailyDialoguePadCollate

def get_LLM_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    testset = LLMDataset(path)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn = LLMPadCollate(dim=0),
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return test_loader

def get_DailyDialogue_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
        
        trainset = DailyDialogueDataset(path, 'train')
        testset = DailyDialogueDataset(path, 'test')
        validset = DailyDialogueDataset(path, 'valid')
    
        train_loader = DataLoader(trainset,
                                batch_size=batch_size,
                                collate_fn = DailyDialoguePadCollate(dim=0),
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        valid_loader = DataLoader(validset,
                                batch_size=batch_size,
                                collate_fn = DailyDialoguePadCollate(dim=0),
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        test_loader = DataLoader(testset,
                                batch_size=batch_size,
                                collate_fn = DailyDialoguePadCollate(dim=0),
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    
        return train_loader, valid_loader, test_loader

def process_data_loader(data, cuda):
    
    input_sequence, qmask, umask, act_labels, emotion_labels, max_sequence_lengths, _ = data
    input_sequence = input_sequence[:, :, :max(max_sequence_lengths)]
    
    if cuda:
        input_sequence, qmask, umask = input_sequence.cuda(), qmask.cuda(), umask.cuda()
        emotion_labels = emotion_labels.cuda()
    
    return [input_sequence, qmask, umask, emotion_labels]


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False, cuda=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        input_sequence, qmask, umask, label = process_data_loader(data, cuda)
        log_prob, alpha, alpha_f, alpha_b = model(input_sequence, qmask, umask)
        
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])      
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='micro', labels=[0,2,3,4,5,6])*100,2)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--checkpoint', type=str, default='dailydialog/best_model.pt', help='path to checkpoint')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    
    n_classes  = 7
    D_e = 100
    D_h = 100
    
    kernel_sizes = [1,2,3]
    
    cuda = args.cuda

    glv_pretrained = np.load(open('dailydialog/glv_embedding_matrix', 'rb'), allow_pickle=True)
    vocab_size, embedding_dim = glv_pretrained.shape

    # load hyperparameters from best_model_config.pkl
    with open('dailydialog/best_model_config.pkl', 'rb') as f:
        config = pickle.load(f)

    model = E2ELSTMModel(D_e, D_h,
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        cnn_output_size=config['cnn_output_size'],
                        cnn_filters=config['cnn_filters'], 
                        cnn_kernel_sizes=config['cnn_kernel_sizes'],
                        cnn_dropout=config['cnn_dropout'],
                        n_classes=n_classes,
                        dropout=config['dropout'],
                        attention=config['attention'])

    model.load_state_dict(torch.load(args.checkpoint))
    
    if cuda:
        model.cuda()
        
    loss_function = MaskedNLLLoss()

    tmp = get_DailyDialogue_loaders('dailydialog/daily_dialogue.pkl', batch_size=config['batch_size'], num_workers=0)
    test_loader = get_LLM_loaders('dailydialog/human_annotation_clean.csv', batch_size=config['batch_size'], num_workers=0)

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    start_time = time.time()
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, cuda=cuda)

    if best_loss == None or best_loss > test_loss:
        best_loss, best_label, best_pred, best_mask, best_attn =\
                test_loss, test_label, test_pred, test_mask, attentions

    # if args.tensorboard:
    #     writer.add_scalar('test: accuracy/loss', test_acc/test_loss, )
    print('test_loss {} test_acc {} test_fscore {} time {}'.\
            format(test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

    # if args.tensorboard:
    #     writer.close()

    print('Test performance..')
    print('Loss {} F1-score {}'.format(best_loss,
                                     round(f1_score(best_label,best_pred,sample_weight=best_mask, average='micro', labels=[0,2,3,4,5,6])*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask,labels=[0,2,3,4,5,6],digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
