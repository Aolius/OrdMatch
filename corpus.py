import os
import json
import torch
import h5py
import numpy as np

def prep_glove(args):
    vocab = {}
    ivocab = []
    tensors = []
    glove_path = os.path.join(args.data_path, 'embedding')
    if not os.path.exists(glove_path):
        os.mkdir(glove_path)
    with open(os.path.join(glove_path, 'glove.840B.300d.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) != 301:
                print(line)
                continue
            assert(len(vals) == 301)
            word = vals[0]
            vec = torch.FloatTensor([ float(v) for v in vals[1:] ])
            vocab[word] = len(ivocab)
            ivocab.append(word)
            tensors.append(vec)
            assert (vec.size(0) == 300)
    assert len(tensors) == len(ivocab)
    tensors = torch.cat(tensors).view(len(ivocab), 300)
    emb_path = os.path.join(glove_path, 'glove_emb.pt')
    with open(emb_path, 'wb') as fpw:
        torch.save([tensors, vocab, ivocab], fpw)


class Dictionary(object):
    def __init__(self, args):
        self.task = args.task
        self.data_dir = args.data_path + '/embedding/'
        self.data_all_dir = args.data_path
        filename = os.path.join(self.data_dir, 'word2idx.pt')
        if os.path.exists(filename):
            self.word2idx = torch.load(os.path.join(self.data_dir,  'word2idx.pt'))
            self.idx2word = torch.load(os.path.join(self.data_dir,  'idx2word.pt'))
            self.word2idx_count = torch.load(os.path.join(self.data_dir,  'word2idx_count.pt'))
        else:
            self.word2idx = {'<<padding>>':0, '<<unk>>':1}
            self.word2idx_count = {'<<padding>>':0, '<<unk>>':0}
            self.idx2word = ['<<padding>>','<<unk>>']

            self.build_dict('train')
            self.build_dict('val')

            torch.save(self.word2idx, os.path.join(self.data_dir,  'word2idx.pt'))
            torch.save(self.idx2word, os.path.join(self.data_dir, 'idx2word.pt'))
            torch.save(self.word2idx_count, os.path.join(self.data_dir,  'word2idx_count.pt'))
        filename_emb = os.path.join(self.data_dir, 'embeddings.pt')
        if os.path.exists(filename_emb):
            self.embs = torch.load(filename_emb)
        else:
            self.embs = self.build_emb(all_vocab=True)

    def build_emb(self, all_vocab=False):
        word2idx = torch.load(os.path.join(self.data_dir, 'word2idx.pt'))
        idx2word = torch.load(os.path.join(self.data_dir,  'idx2word.pt'))
        emb = torch.FloatTensor(len(idx2word), 300).zero_()
        print('loading Glove ...')
        print('Raw vocabulary size: ', str(len(idx2word)))

        if not os.path.exists(os.path.join(self.data_dir, 'glove_emb.pt')): prep_glove()
        glove_tensors, glove_vocab, glove_ivocab = torch.load(os.path.join(self.data_dir, 'glove_emb.pt'))
        if not all_vocab:
            self.word2idx = {'<<padding>>': 0, '<<unk>>': 1}
            self.idx2word = ['<<padding>>', '<<unk>>']
        count = 0
        for w_id, word in enumerate(idx2word):
            if word in glove_vocab:
                id = self.add_word(word)
                emb[id] = glove_tensors[glove_vocab[word]]
                count += 1
            else:
                id = self.add_word(word)
                emb[id] = torch.FloatTensor(np.random.normal(0, 0.01, size=300))
       
        emb = emb[:len(self.idx2word)]
        print("Number of words not appear in glove: ", len(idx2word) - count)
        print("Vocabulary size: ", len(self.idx2word))

        torch.save(emb, os.path.join(self.data_dir, 'embeddings.pt'))
        torch.save(self.word2idx, os.path.join(self.data_dir, 'word2idx.pt'))
        torch.save(self.idx2word, os.path.join(self.data_dir, 'idx2word.pt'))

        return emb


    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.word2idx_count[word] = 1
        else:
            self.word2idx_count[word] += 1

        return self.word2idx[word]

    def build_dict(self, dataset):

        #use TC's vocabulary for all tasks, since AO8/AO24 are directly generated from TC
        filename = os.path.join(self.data_all_dir,f'TC_{dataset}.json')
        assert (os.path.exists(filename))
        with open(filename, 'r', encoding='utf-8') as fpr:
            data_all = json.load(fpr)
            for instance in data_all:
                    words = [word for title in instance['question'] for word in title]
                    for option in instance['options']: words += option
                    for step in instance['passage']: words += step
                    for word in words: self.add_word(word)

    def __len__(self):
        return len(self.idx2word)




class Corpus(object):
    def __init__(self, args):
        self.task = args.task
        if self.task not in ['TC', 'AO24', 'AO8']:
             assert False, 'the task' + self.task + ' is not supported!'
        self.dictionary = Dictionary(args)
        self.data_dir = args.data_path


        self.data_all, self.start_id, self.indices = {}, {}, {}

        #load textual cloze (TC) data
        setnames = ['train', 'val']
        for setname in setnames:
            self.data_all[setname] = self.load_data(os.path.join(self.data_dir, f'TC_{setname}.json'))
            print(setname, len(self.data_all[setname]))
            self.start_id[setname] = 0
            self.indices[setname] = torch.randperm((len(self.data_all[setname]))) if setname == 'train' else torch.range(0,len(self.data_all[setname]))
        #load activity ordering (AO8/AO24) data
        setnames_a = ['AO24_train', 'AO24_val', 'AO8_train', 'AO8_val']
        for setname_a in setnames_a:
            self.data_all[setname_a] = self.load_data(os.path.join(self.data_dir, f'{setname_a}.json'))
            print(setname_a, len(self.data_all[setname_a]))
            self.start_id[setname_a] = 0
            self.indices[setname_a] = torch.randperm(
                (len(self.data_all[
                         setname_a]))) if setname_a == 'AO24_train' or setname_a == 'AO8_train' else torch.range(0, len(
                self.data_all[setname_a]))



    def get_batch(self, batch_size, setname, div=True):
        if self.start_id[setname] >= len(self.data_all[setname]):
            self.start_id[setname] = 0
            if setname == 'train': self.indices[setname] = torch.randperm(len(self.data_all[setname]))

        end_id = self.start_id[setname] + batch_size if self.start_id[setname] + batch_size < len(self.data_all[setname]) else len(self.data_all[setname])
        documents, questions, answers, qa_pairs, qa_pairs_div, labels = [], [], [], [], [], []
        for i in range(self.start_id[setname], end_id):
            instance_id = int(self.indices[setname][i])
            instance = self.data_all[setname][instance_id]
            qa_pairs.append(instance['qa_pairs'])
            if div:
                qa_pairs_div.append(instance['qa_pairs_div'])
            # questions.append(instance['question'])
            answers.append(instance['options'])
            documents.append(instance['passage'])
            labels.append(instance['ground_truth'])

        self.start_id[setname] += batch_size

        documents = self.seq2Dtensor(documents)
        qa_pairs = self.seq2QAtensor(qa_pairs)
        if div:
            qa_pairs_div = self.seq2QADtensor(qa_pairs_div)
      #  ans = self.seq2Atensor(answers)
        labels = torch.LongTensor(labels)
        if div:
            return [documents, qa_pairs_div, labels]
        return [documents, qa_pairs, labels]

    def get_batch_attack(self,batch_size,setname='AO24_train',div=True):
        if self.start_id[setname] >= len(self.data_all[setname]):
            self.start_id[setname] = 0
            if setname == 'AO24_train' or setname == 'AO8_train': self.indices[setname] = torch.randperm(len(self.data_all[setname]))

        end_id = self.start_id[setname] + batch_size if self.start_id[setname] + batch_size < len(
            self.data_all[setname]) else len(self.data_all[setname])
        documents, questions, answers, qa_pairs, qa_pairs_div, labels = [], [], [], [], [], []
        for i in range(self.start_id[setname], end_id):
            instance_id = int(self.indices[setname][i])
            instance = self.data_all[setname][instance_id]
            qa_pairs.append(instance['qa_shuffle'])
            if div:
                qa_pairs_div.append(instance['qa_divs'])

            # questions.append(instance['question'])

            documents.append(instance['passage'])
            labels.append(instance['ground_truth'])



        self.start_id[setname] += batch_size

        documents = self.seq2Dtensor(documents)
        if setname == 'AO24_train' or setname == 'AO24_val':
            qa_pairs = self.seq2QAtensor(qa_pairs,option_num=24)
            if div:
                qa_pairs_div = self.seq2QADtensor(qa_pairs_div)
        else:
            qa_pairs = self.seq2QAtensor(qa_pairs, option_num=8)
            if div:
                qa_pairs_div = self.seq2QADtensor(qa_pairs_div,num_options=8)


        #  ans = self.seq2Atensor(answers)
        labels = torch.LongTensor(labels)
        if div:
            return [documents, qa_pairs_div, labels]
        return [documents, qa_pairs, labels]








    def seq2Atensor(self, docs, a_len_bound = 5):

        a_len_max = max([len(a) for doc in docs for a in doc ])
        a_len_max = min(a_len_max, a_len_bound)

        a_tensor = torch.LongTensor(len(docs), 4, a_len_max).zero_()
        a_len = torch.LongTensor(len(docs),4).zero_()

        for d_id, doc in enumerate(docs):
            for a_id, a in enumerate(doc):
                a_len[d_id][a_id] = len(a)
                for w_id, word in enumerate(a):
                    if w_id >= a_len_max: break
                    a_tensor[d_id][a_id][w_id] = self.dictionary.word2idx.get(word, 1)


        return [a_tensor, a_len]
    def seq2QAtensor(self, docs, qa_len_bound = 20, option_num=4):
        '''

        :param docs: list of size (batch_size, 4, qa_length)
        :param qa_len_bound: limit of word number of the qa pair
        :return:
        '''
        qa_len_max = max([len(qa) for doc in docs for qa in doc ])
        qa_len_max = min(qa_len_max, qa_len_bound)

        qa_tensor = torch.LongTensor(len(docs), option_num, qa_len_max).zero_()
        qa_len = torch.LongTensor(len(docs),option_num).zero_()

        for d_id, doc in enumerate(docs):
            for qa_id, qa in enumerate(doc):
                qa_len[d_id][qa_id] = len(qa)
                for w_id, word in enumerate(qa):
                    if w_id >= qa_len_max: break
                    qa_tensor[d_id][qa_id][w_id] = self.dictionary.word2idx.get(word, 1)

        return [qa_tensor, qa_len]
    def seq2QADtensor(self, docs, num_options=24, qa_len_bound = 5):
        '''

        :param docs: list of size (batch_size, 4, 4, qa_phrase_length)
        :param qa_len_bound: limit of word number of the qa pair
        :return:
        '''
        # qa_len_max = max([len(qa) for doc in docs for qa in doc ])
        # qa_len_max = min(qa_len_max, qa_len_bound)
        qa_len_max = qa_len_bound

        qa_tensor = torch.LongTensor(len(docs), num_options, 4, qa_len_max).zero_()
        qa_len = torch.LongTensor(len(docs),num_options, 4).zero_()

        for d_id, doc in enumerate(docs):
            for qa_id, qa in enumerate(doc):
                for phrase_id, phrase in enumerate(qa):
                    qa_len[d_id][qa_id][phrase_id] = len(phrase)
                    for w_id, word in enumerate(phrase):
                        if w_id >= qa_len_max: break
                        qa_tensor[d_id][qa_id][phrase_id][w_id] = self.dictionary.word2idx.get(word, 1)


        return [qa_tensor, qa_len]
    def seq2Dtensor(self, docs, img_list=None, step_num_bound=25, step_len_bound=50, whichset='train'):
        '''

        :param docs:  list of size (batch_size, step_num, step_length)
        :param step_num_bound: limit of step number of a recipe
        :param step_len_bound: limit of word number of a step
        :return:
        '''
        step_num_max = max([len(doc) for doc in docs])
        step_num_max = min(step_num_max, step_num_bound)
        step_len_max = max([len(step) for doc in docs for step in doc])
        step_len_max = min(step_len_max, step_len_bound)

        step_tensor = torch.LongTensor(len(docs), step_num_max, step_len_max).zero_()
        step_len = torch.LongTensor(len(docs), step_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()



        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, step in enumerate(doc):
                if s_id >= step_num_max:
                    break
                step_len[d_id][s_id] = len(step)
                for w_id, word in enumerate(step):
                    if w_id >= step_len_max:
                        break
                    #if word does not exist in dictionary, set it to 1
                    step_tensor[d_id][s_id][w_id] = self.dictionary.word2idx.get(word, 1)


        if img_list:
            if whichset == 'train':
                img_step_tensor = torch.Tensor(len(docs), step_num_max, 49, 2048).zero_()
                for d_id, doc in enumerate(img_list):
                    for step_id, step in enumerate(doc):
                        if len(step) < 1: continue
                        img_idx = self.train_img2idx[step[0]]
                        img_feat = self.train_img[img_idx]
                        img_step_tensor[d_id][step_id] = torch.from_numpy(img_feat)
            elif whichset == 'val':
                img_step_tensor = torch.Tensor(len(docs), step_num_max, 49, 2048).zero_()
                for d_id, doc in enumerate(img_list):
                    for step_id, step in enumerate(doc):
                        if len(step) < 1: continue
                        img_idx = self.val_img2idx[step[0]]
                        img_feat = self.val_img[img_idx]
                        img_step_tensor[d_id][step_id] = torch.from_numpy(img_feat)


            return img_step_tensor, [step_tensor, doc_len, step_len]
        else:
            return step_tensor, doc_len, step_len








    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as fpr:
            data = json.load(fpr)
        return data
