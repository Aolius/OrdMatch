import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from ordmatch import OrdMatch

from corpus import Corpus
from config import args
from evaluate import evaluation, accuracy

args.task = 'AO24'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.set_num_threads(3)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


corpus = Corpus(args)
print("Corpus built.")
model = OrdMatch(corpus, args)
model.train()
criterion = nn.NLLLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adamax(parameters, lr=args.lr)


if args.cuda:
    model.cuda()
    criterion.cuda()

start_time = time.time()
total_loss = 0
total_loss1 = 0
total_loss2 = 0

interval = args.interval
save_interval = len(corpus.data_all['train']) // args.batch_size

best_dev_score = -99999
iterations = args.epochs*len(corpus.data_all['train']) // args.batch_size
print('max iterations: '+str(iterations))

for iter in range(iterations):
    optimizer.zero_grad()
    data = corpus.get_batch_attack(args.batch_size, 'train',div=True)
    output1, output2, att = model(data)


    labels = data[2].cuda() if args.cuda else data[2]
    _, pred = output1.max(1)
    score = accuracy(labels, pred)
    loss1 = criterion(output1, labels)
    loss2 = criterion(output2, labels)
    loss = (1 - args.lamda) * loss1 + args.lamda * loss2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    total_loss += float(loss.data)
    total_loss1 += float(loss1.data)
    total_loss2 += float(loss2.data)

    if iter % interval == 0:
        cur_loss = total_loss / interval if iter!=0 else total_loss
        cur_loss1 = total_loss1 / interval if iter != 0 else total_loss1
        cur_loss2 = total_loss2 / interval if iter != 0 else total_loss2
        elapsed = time.time() - start_time
        print('| iterations {:3d} | start_id {:3d} | ms/batch {:5.2f} | loss {:5.3f} loss1 {:5.3f} loss2 {:5.3f}   '.format(
        iter, corpus.start_id['train'], elapsed * 1000 / interval, cur_loss, cur_loss1, cur_loss2))
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0


        start_time = time.time()

    if iter % save_interval == 0:
        save_path = os.path.join(args.save_path, args.task)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save([model, optimizer, criterion], os.path.join(save_path, f'save_24_{args.lamda}.pt'))
        score = evaluation(model, corpus, args.task, args.batch_size, dataset='val', div=True, reg=True)
        print('DEV accuracy: ' + str(score))
        with open(os.path.join(save_path, f'record_24_{args.lamda}.txt'), 'a', encoding='utf-8') as fpw:
            if iter == 0: fpw.write(str(args) + '\n')
            fpw.write(str(iter) + ':\tDEV accuracy:\t' + str(score) + '\n')

        if score > best_dev_score:
            best_dev_score = score
            torch.save([model, optimizer, criterion], os.path.join(save_path, f'save_best_24_{args.lamda}.pt'))

    # if (iter+1) % (len(corpus.data_all['train']) // args.batch_size) == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.95


# model, optimizer, criterion = torch.load('/hdd/liuao/recipe-data/data/trainedmodel/'+args.task+'_save_best_v2.pt')
# score = evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size, dataset='test')
# with open('/hdd/liuao/recipe-data/data/trainedmodel/'+args.task+'_record_v2.txt', 'a', encoding='utf-8') as fpw:
#     fpw.write('TEST accuracy:\t' + str(score) + '\n')
# print('TEST accuracy: ' + str(score))
