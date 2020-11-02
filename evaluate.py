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
