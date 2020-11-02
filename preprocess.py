import glob
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import random
from config import args

def sent_word_tokenize(text):
    sents = sent_tokenize(text)
    words = [word_tokenize(s) for s in sents]
    return words

def data_shuffle():
    permutations = [[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],
                    [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],
                    [3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],
                    [4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
    dataset_names = ['train', 'val']
    for dataset_name in dataset_names:
        data_all = []
        data_path = f'{args.data_path}/{dataset_name}.json'
        output_data_path = os.path.join(args.data_path, f'AO24_{dataset_name}.json')
        if os.path.exists(output_data_path):
            print(f'file {output_data_path} already exists!')
            break
        if not os.path.exists(data_path): break
        with open(data_path,'r',encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data = data_raw['data']
                data_size = 0
                for _, recipe in enumerate(tqdm(data)):
                    if recipe['task'] == 'textual_cloze':
                        instance = {}
                        instance['ground_truth_o'] = recipe['answer']
                        instance['options'] = [word_tokenize(option.lower().strip()) for option in recipe['choice_list']]
                        instance['placeholder'] = -1
                        for i, title in enumerate(recipe['question']):
                            if (title.startswith('@')):
                                data_size += 1
                                instance['placeholder'] = i
                        instance['question'] = [word_tokenize(q_title.lower().strip()) for q_title in recipe['question']]

                        qa_shuffle = []
                        qa_right = instance['question'][:]
                        qa_right[instance['placeholder']] = instance['options'][instance['ground_truth_o']]
                        qa_shuffle.append(qa_right)
                        shuffle_orders = random.sample(range(0,len(permutations)),3) #select 3 wrong orders
                        for order in shuffle_orders: #for each order
                            qa_ = []
                            for i in permutations[order]: #for each index in order
                                qa_.append(qa_right[i-1])
                            qa_shuffle.append(qa_)
                        qa_shuffle_question = qa_shuffle[:]
                        correct_pos = random.randint(0,3) #decide which position is the correct answer
                        qa_shuffle_question[correct_pos] = qa_shuffle[0]
                        j = 0
                        for i in range(3):
                            if not i == correct_pos:
                                qa_shuffle_question[i] = qa_shuffle[1:][j]
                                j = j + 1
                            else:
                                continue
                        instance['ground_truth'] = correct_pos
                        instance['qa_shuffle'] = []
                        for i in range(4):
                            choice = qa_shuffle_question[i]
                            flatted = []
                            for j in range(4):
                                flatted.extend(choice[j])
                            instance['qa_shuffle'].append(flatted)



                        steps = recipe['context']
                        passage = []
                        images = []
                        p_length = 0
                        for step in steps:
                            passage.append(word_tokenize(step['body'].lower().strip()))
                            images.append(step['images'])
                            p_length += 1
                        instance['passage'] = passage
                        instance['image_list'] = images
                        instance['step_size'] = p_length
                        data_all.append(instance)
                print(f'{dataset_name} data size for Activity Ordering24 task:{data_size}')

        with open(output_data_path, 'w', encoding='utf-8') as fout:
                json.dump(data_all, fout, indent=4)





def preprocess(task='textual_cloze', args):
    print('Preprocessing the dataset ' + task + '...')
    dataset_names = ['train', 'val']
    for dataset_name in dataset_names:
        data_all = []
        data_path = f'{args.data_path}/{dataset_name}.json'
        if not os.path.exists(data_path): break
        output_data_path = os.path.join(args.data_path, f'{task}_{dataset_name}.json')
        if os.path.exists(output_data_path):
            print(f'file {output_data_path} already exists!')
            break
        with open(data_path,'r',encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data = data_raw['data']
                data_size = 0
                for recipe in data:
                    if recipe['task'] == task:
                        if task == 'textual_cloze':
                            instance = {}
                            instance['ground_truth'] = recipe['answer']
                            instance['options'] = [word_tokenize(option.lower()) for option in recipe['choice_list']]
                            instance['placeholder'] = -1
                            for i, title in enumerate(recipe['question']):
                                if (title.startswith('@')):
                                    data_size += 1
                                    instance['placeholder'] = i
                            instance['question'] = [word_tokenize(q_title.lower()) for q_title in recipe['question']]

                            # create 4 triplets (q,a_i) for each a_i in option list, the form of which is [words]
                            instance['qa_pairs'] = []
                            for i in range(4):
                                qa = instance['question'][:]
                                qa[instance['placeholder']] = instance['options'][i]
                                flatted_qa = []
                                for title in qa:
                                    flatted_qa.extend(title)
                                instance['qa_pairs'].append(flatted_qa)
                            instance['qa_pairs_div'] = []
                            for i in range(4):
                                qa = instance['question'][:]
                                qa[instance['placeholder']] = instance['options'][i]
                                flatted_qa = []
                                for title in qa:
                                    flatted_qa.append(title)
                                instance['qa_pairs_div'].append(flatted_qa)


                            steps = recipe['context']
                            passage = []
                            images = []
                            p_length = 0
                            for step in steps:
                                passage.append(word_tokenize(step['body'].lower().strip()))
                                images.append(step['images'])
                                p_length += 1
                            instance['passage'] = passage
                            instance['image_list'] = images
                            instance['step_size'] = p_length
                            data_all.append(instance)
                        elif task == 'visual_cloze':
                            instance = {}
                            instance['ground_truth'] = recipe['answer']
                            instance['options'] = recipe['choice_list']
                            instance['placeholder'] = -1
                            for i, title in enumerate(recipe['question']):
                                if (title.startswith('@')):
                                    data_size += 1
                                    instance['placeholder'] = i
                            instance['question'] = recipe['question']

                            # create 4 triplets (q,a_i) for each a_i in option list, the form of which is [words]
                            instance['qa_pairs'] = []
                            for i in range(4):
                                qa = instance['question']
                                qa[instance['placeholder']] = instance['options'][i]
                                instance['qa_pairs'].append(qa)

                            steps = recipe['context']
                            passage = []
                            titles = []
                            p_length = 0
                            for step in steps:
                                passage.append(word_tokenize(step['body'].strip()))
                                titles.append(word_tokenize(step['title'].strip()))
                                p_length += 1
                            instance['passage'] = passage
                            instance['titles'] = titles
                            instance['step_size'] = p_length
                            data_all.append(instance)
                        elif task == 'visual_coherence':
                            instance = {}
                            data_size += 1
                            instance['ground_truth'] = recipe['answer']
                            instance['options'] = recipe['choice_list']
                            instance['question'] = recipe['question']
                            steps = recipe['context']
                            passage = []
                            titles = []
                            p_length = 0
                            for step in steps:
                                passage.append(word_tokenize(step['body'].strip()))
                                titles.append(word_tokenize(step['title'].strip()))
                                p_length += 1
                            instance['passage'] = passage
                            instance['titles'] = titles
                            instance['step_size'] = p_length
                            data_all.append(instance)
                        elif task == 'visual_ordering':
                            instance = {}
                            data_size += 1
                            instance['ground_truth'] = recipe['answer']
                            instance['options'] = recipe['choice_list']

                            steps = recipe['context']
                            passage = []
                            titles = []
                            p_length = 0
                            for step in steps:
                                passage.append(word_tokenize(step['body'].strip()))
                                titles.append(word_tokenize(step['title'].strip()))
                                p_length += 1
                            instance['passage'] = passage
                            instance['titles'] = titles
                            instance['step_size'] = p_length
                            data_all.append(instance)
                        else:
                            assert False, 'the task ' + task + ' is not supported!'




                print(f'{dataset_name} data size for task {task}:{data_size}')


        with open(output_data_path, 'w', encoding='utf-8') as fout:
            json.dump(data_all, fout)

if __name__ == '__main__':
    #preprocess original data file to get train/val data file for textual cloze task
    preprocess('textual_cloze',args)
    #shuffle original data file to create train/val data file for activity ordering task
    data_shuffle()














