'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch


def accuracy(ground_truth, prediction):
    assert(len(ground_truth) == len(prediction))
    accuracy = float( (ground_truth==prediction).float().mean(0) )
    return accuracy

def evaluation(model, corpus, task, batch_size, dataset='val',div=False, reg=False):

    model.eval()
    labels_all = []
    pred_all = []

    while True:
        if task == 'AO8' or task == 'AO24':
            data = corpus.get_batch_attack(batch_size,dataset,div=div)
        else:
            data = corpus.get_batch(batch_size, dataset, div=div)

        if reg:
            #reg=True indicates full ordMatch model
            output1, _, _ = model(data)
            _, pred = output1.max(1)
        else:
            #reg=False indicates only matching model
            output, _ = model(data)
            _, pred = output.max(1)

        pred_all.append(pred.cpu())
        labels_all.append(data[2])

        if corpus.start_id[dataset] >= len(corpus.data_all[dataset]): break
    score = accuracy( torch.cat(labels_all), torch.cat(pred_all) )


    model.train()
    return score
