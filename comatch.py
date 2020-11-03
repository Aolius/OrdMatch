import torch
import torch.nn as nn
from torch.autograd import Variable
from self_attention_block import MultiHeadAttention
from cnn_classifier import CNN_Classifier


from torch.nn.utils.weight_norm import weight_norm
def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i,:,:seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        #重新mask一下，这时该维度上和不为1了，因此再除以sum重新归一一下
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result

class MatchNet(nn.Module):
    def __init__(self, mem_dim, dropoutP):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(2*mem_dim, 2*mem_dim)
        self.trans_linear = nn.Linear(mem_dim, mem_dim)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs

        #proj_p: H^p
        #proj_q: H^a
        #trans_q: W^gH^a + b^g

        elem_min = proj_q - proj_p
        elem_mul = proj_q * proj_p
        all_con = torch.cat([elem_min,elem_mul], 2)
        #all_con: (10*4, 23*50, 600).


        output = nn.ReLU()(self.map_linear(all_con))

        return output

class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.3):
        super(MaskLSTM, self).__init__()
        #注意这里是双向LSTM因此输出的实际dim为2倍的out_dim也即300
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs, return_y=False):
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i,:seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input*mask_in)

        H, (hn,cn) = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i,:seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask
        if return_y: return output[:,-1]
        return output


class CoMatch_cross_attention(nn.Module):
    def __init__(self,corpus, args):
        super(CoMatch_cross_attention,self).__init__()
        self.emb_dim = 300
        self.hid_dim = args.hid_dim
        self.dropoutP = args.dropoutP
        self.use_cuda = args.cuda

        self.embs = nn.Embedding(len(corpus.dictionary), self.emb_dim)
        self.embs.weight.data.copy_(corpus.dictionary.embs)
        self.embs.weight.requires_grad = False

        self.encoder = MaskLSTM(self.emb_dim, self.hid_dim, dropoutP=self.dropoutP)
        self.l_encoder = MaskLSTM(self.hid_dim*4, self.hid_dim, dropoutP=self.dropoutP)
        self.h_encoder = MaskLSTM(self.hid_dim*2, self.hid_dim, dropoutP=self.dropoutP)
        self.d_qa_att = MultiHeadAttention()
        self.qa_d_att = MultiHeadAttention()
        self.cnn = CNN_Classifier()
        self.out_score = nn.Linear(300, 1)
        self.match_module = MatchNet(self.hid_dim*2, self.dropoutP)
        self.rank_module = nn.Linear(self.hid_dim*2, 1)

        self.drop_module = nn.Dropout(self.dropoutP)


    def generate_mask(self, len_1, len_2):

        mask_1 = len_1.new_zeros(len_1.size()).unsqueeze(1).repeat(1,50)
        for i in range(len_1.size(0)):
            if i >= 50:
                break
            mask_1[i, :len_1[i]] = 1

        mask_2 = len_2.new_zeros(len_2.size()).unsqueeze(1).repeat(1,50)
        for i in range(len_2.size(0)):
            if i >=50:
                break
            mask_2[i, :len_2[i]] = 1

        mask = torch.einsum('bi,bj->bij',(mask_1, mask_2))

        return mask.byte()


    # def forward(self, inputs):
    #     docs, qas, labels = inputs
    #     d_word, d_len, d_s_len = docs
    #     qa_word, qa_len = qas
    #
    #     if self.use_cuda:
    #         d_word, d_len, d_s_len, qa_word, qa_len = d_word.cuda(), d_len.cuda(), d_s_len.cuda(), qa_word.cuda(), qa_len.cuda()
    #
    #     # document: d_embs.shape = (batch_size, sent_num, sent_len, emb_dim)
    #     d_embs = self.drop_module(Variable(self.embs(d_word), requires_grad=True))
    #     # qa_pair: qa_embs.shape = (batch_size, option_num, sent_len, emb_dim)
    #     qa_embs = self.drop_module(Variable(self.embs(qa_word), requires_grad=True))
    #
    #     # cross attention
    #     d_qa_scores = []
    #     for i, qa_emb in enumerate(qa_embs.unbind(1)):
    #         d_tmp = []
    #         for j, d_emb in enumerate(d_embs.unbind(1)):
    #             d_qa_mask = self.generate_mask(d_s_len[:,j], qa_len[:,i])
    #             qa_d_mask = self.generate_mask(qa_len[:,i], d_s_len[:,j])
    #             # d_qa,qa_d.shape = (batch_size, sent_len, emb_dim)
    #             d_qa, d_qa_att = self.m_att(query=d_emb, key=qa_emb, value=qa_emb, attn_mask=d_qa_mask)
    #             qa_d, qa_d_att = self.m_att(query=qa_emb, key=d_emb, value=d_emb, attn_mask=qa_d_mask)
    #
    #             # d_qa_match.shape = (batch_size, sent_len, sent_len)
    #             d_qa_match = torch.einsum('bik,bjk->bij',(d_qa, qa_d)) / np.sqrt(300)
    #             # d_qa_feature.shape = (batch_size, emb_dim)
    #             d_qa_feature = self.cnn(d_qa_match.unsqueeze(1))
    #             d_tmp.append(d_qa_feature)
    #         d_tmp, _ = torch.max(torch.stack(d_tmp, 0), 0)
    #         score = self.out_score(d_tmp)
    #         d_qa_scores.append(score)
    #     d_qa_scores = torch.stack(d_qa_scores, 1)
    #
    #     output = torch.nn.functional.log_softmax(d_qa_scores.squeeze(2))
    #
    #     return output

    def forward(self, inputs):
        docs, qas, labels = inputs
        d_word, d_len, d_s_len = docs
        qa_word, qa_len = qas

        if self.use_cuda:
            d_word, d_len, d_s_len, qa_word, qa_len = d_word.cuda(), d_len.cuda(), d_s_len.cuda(), qa_word.cuda(), qa_len.cuda()

        # document: d_embs.shape = (batch_size, sent_num, sent_len, emb_dim)
        d_embs = self.drop_module(Variable(self.embs(d_word), requires_grad=True))
        # qa_pair: qa_embs.shape = (batch_size, option_num, sent_len, emb_dim)
        qa_embs = self.drop_module(Variable(self.embs(qa_word), requires_grad=True))

        # cross attention
        score = []
        for i, qa_emb in enumerate(qa_embs.unbind(1)):
            tmp = []
            for j, d_emb in enumerate(d_embs.unbind(1)):
                d_qa_mask = self.generate_mask(d_s_len[:,j], qa_len[:,i])
                qa_d_mask = self.generate_mask(qa_len[:,i], d_s_len[:,j])
                # d_qa,qa_d.shape = (batch_size, sent_len, emb_dim)
                d_qa, d_qa_att = self.d_qa_att(query=d_emb, key=qa_emb, value=qa_emb, attn_mask=d_qa_mask)
                qa_d, qa_d_att = self.qa_d_att(query=qa_emb, key=d_emb, value=d_emb, attn_mask=qa_d_mask)

                # out_dim = 600
                d_qa_match = self.match_module([d_qa, qa_d, qa_len.view(-1)])

                l_hidden = self.l_encoder([d_qa_match, d_s_len[:,j]])
                l_hidden_pool, _ = l_hidden.max(1)
                tmp.append(l_hidden_pool)
            tmp = torch.stack(tmp, 1)

            h_hidden = self.h_encoder([tmp,d_len])
            h_hidden_pool, _ = h_hidden.max(1)
            h_score = self.rank_module(h_hidden_pool).squeeze(1)
            score.append(h_score)
        score = torch.stack(score, 1)

        output = torch.nn.functional.log_softmax(score)


        return output

class Attention(nn.Module):
    def __init__(self,v_dim, d_dim, num_hid):
        super(Attention, self).__init__()

        self.linear1 = weight_norm(nn.Linear(v_dim+d_dim, num_hid))
        self.linear2 = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, d):
        '''

        :param v: (batch*step, 49, 2048)
        :param d: (batch*step, 300)
        :return:
        '''
        num_objs = v.size(1)
        #d: (batch*step, 49, 300)

        d = d.unsqueeze(1).repeat(1,num_objs,1)

        #vd: (batch*step, 49, 2348)
        vd = torch.cat((v,d),2)
        #joint_repr: (_, 49, num_hid)
        joint_repr = nn.functional.relu(self.linear1(vd))
        #logits: (_, 49, 1)
        logits = self.linear2(joint_repr)
        #w: (TODO)
        w = nn.functional.softmax(logits, 1)
        return w


class ImgMatch(nn.Module):
    def __init__(self,corpus, args):
        super(ImgMatch,self).__init__()
        self.emb_dim = 300
        self.hid_dim = args.hid_dim
        self.dropoutP = args.dropoutP
        self.use_cuda = args.cuda
        self.embs = nn.Embedding(len(corpus.dictionary), self.emb_dim)
        self.embs.weight.data.copy_(corpus.dictionary.embs)
        self.embs.weight.requires_grad = False
        self.att_encoder = MaskLSTM(self.emb_dim, self.hid_dim*2, bidirectional=False, dropoutP=self.dropoutP)
        self.encoder = MaskLSTM(self.emb_dim, self.hid_dim, dropoutP=self.dropoutP)
        self.l_encoder = MaskLSTM(self.hid_dim*4, self.hid_dim, dropoutP=self.dropoutP)
        self.h_encoder = MaskLSTM(self.hid_dim*2, self.hid_dim, dropoutP=self.dropoutP)
        self.match_module = MatchNet(self.hid_dim*2, self.dropoutP)
        self.rank_module = nn.Linear(self.hid_dim*2, 1)
        self.t2iAtt = Attention(2048, 300, 1024)
        self.drop_module = nn.Dropout(self.dropoutP)

        self.img2word = nn.Linear(2048,300)
        # self.data_dir = '/hdd/liuao/recipe-data/data/images'
        #
        # model_name = 'resnet152'
        # self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # self.model.cuda()
        # self.model.eval()
        # self.load_img = utils.LoadImage()
        #
        # self.tf_img = utils.TransformImage(self.model)

    def forward(self, inputs):

        docs, qas, labels, img_feat = inputs
        d_word, d_len, d_s_len = docs
        qa_word, qa_len = qas

       # if self.use_cuda:
        d_word, d_len, d_s_len, qa_word, qa_len,img_feat= d_word.cuda(), d_len.cuda(), d_s_len.cuda(), qa_word.cuda(), qa_len.cuda(),img_feat.cuda()



        d_embs = self.drop_module(Variable(self.embs(d_word), requires_grad=False))
        qa_embs = self.drop_module(Variable(self.embs(qa_word), requires_grad=False))

        d_hidden_for_att = self.att_encoder([d_embs.view(d_embs.size(0) * d_embs.size(1), d_embs.size(2), self.emb_dim), d_s_len.view(-1)], return_y=True)
        img_reshape = img_feat.view(img_feat.size(0)*img_feat.size(1),img_feat.size(2), img_feat.size(3))

        t2i_att = self.t2iAtt(img_reshape, d_hidden_for_att)
        #img_att : (batch*step, 2048)
        img_att = (t2i_att * img_reshape).sum(1)
        img_hidden = self.img2word(img_att)


        d_hidden = self.encoder([d_embs.view(d_embs.size(0)*d_embs.size(1), d_embs.size(2), self.emb_dim), d_s_len.view(-1)])
        qa_hidden = self.encoder([qa_embs.view(qa_embs.size(0)*qa_embs.size(1), qa_embs.size(2), self.emb_dim), qa_len.view(-1)])

        d_hidden = torch.cat((img_hidden.unsqueeze(1),d_hidden), 1)


        d_hidden_3d = d_hidden.view(d_embs.size(0), d_embs.size(1)*(d_embs.size(2)+1), d_hidden.size(-1))
        d_hidden_3d_repeat = d_hidden_3d.repeat(1, qa_embs.size(1), 1).view(d_hidden_3d.size(0)*qa_embs.size(1), d_hidden_3d.size(1), d_hidden_3d.size(2))

        #out_dim = 600
        d_qa_match = self.match_module([d_hidden_3d_repeat, qa_hidden, qa_len.view(-1)])
      #  print("d_qa_match: ", d_qa_match.size())
        co_match_hier = d_qa_match.view(d_embs.size(0)*qa_embs.size(1)*d_embs.size(1), d_embs.size(2)+1, -1)
      #  print("comatchhier: ", co_match_hier.size())

        l_hidden = self.l_encoder([co_match_hier, d_s_len.repeat(1, qa_embs.size(1)).view(-1)])
        l_hidden_pool, _ = l_hidden.max(1)

        h_hidden = self.h_encoder([l_hidden_pool.view(d_embs.size(0)*qa_embs.size(1), d_embs.size(1), -1), d_len.view(-1,1).repeat(1, qa_embs.size(1)).view(-1)])
        h_hidden_pool, _ = h_hidden.max(1)

        o_rep = h_hidden_pool.view(d_embs.size(0), qa_embs.size(1), -1)
        output = torch.nn.functional.log_softmax( self.rank_module(o_rep).squeeze(2))

        return output