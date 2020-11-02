import torch
import torch.nn as nn
from torch.autograd import Variable


def masked_softmax(att, p_len, max_p_len, q_len, max_q_len, mask_value=-2 ** 32 + 1):

    p_len_ = p_len.view(-1)
    p_mask = torch.zeros_like(p_len_).unsqueeze(-1).expand(size=[p_len_.size(0), max_p_len]).clone()
    for i in range(p_len_.size(0)):
        p_mask[i,:p_len_[i]] = 1
        # print(np.array(p_mask[i,:]).sum(), np.array(p_len_[i]))
    p_mask =  p_mask.view(-1, p_len.size(-1), max_p_len)

    q_mask = torch.zeros_like(q_len).unsqueeze(-1).expand(size=[q_len.size(0), max_q_len]).clone()
    for i in range(q_len.size(0)):
        q_mask[i, :q_len[i]] = 1
        # print(np.array(q_mask[i,:]).sum(), np.array(q_len[i]))

    mask = torch.einsum("bim,bn->bimn", (p_mask,q_mask)).float()

    p_result = torch.nn.functional.softmax(att * mask + mask_value * (1-mask), dim=-1)
    q_result = torch.nn.functional.softmax(att * mask + mask_value * (1 - mask), dim=-2)

    return p_result, q_result

def att_mask(p_len, max_p_len, q_len, max_q_len, mask_value=-2 ** 32 + 1):

    p_mask = torch.zeros_like(p_len).unsqueeze(-1).expand(size=[p_len.size(0), max_p_len]).clone()
    for i in range(p_len.size(0)):
        p_mask[i,:p_len[i]] = 1
        # print(np.array(p_mask[i,:]).sum(), np.array(p_len_[i]))

    q_mask = torch.zeros_like(q_len).unsqueeze(-1).expand(size=[q_len.size(0), max_q_len]).clone()
    for i in range(q_len.size(0)):
        q_mask[i, :q_len[i]] = 1
        # print(np.array(q_mask[i,:]).sum(), np.array(q_len[i]))

    mask = torch.einsum("bm,bn->bmn", (p_mask,q_mask)).float()

    return mask


def masked_softmax_2(att, p_len, max_p_len, q_len, max_q_len, mask_value=-2 ** 32 + 1):

    p_mask = torch.zeros_like(p_len).unsqueeze(-1).expand(size=[p_len.size(0), max_p_len]).clone()
    for i in range(p_len.size(0)):
        p_mask[i,:p_len[i]] = 1
        # print(np.array(p_mask[i,:]).sum(), np.array(p_len_[i]))

    q_mask = torch.zeros_like(q_len).unsqueeze(-1).expand(size=[q_len.size(0), max_q_len]).clone()
    for i in range(q_len.size(0)):
        q_mask[i, :q_len[i]] = 1
        # print(np.array(q_mask[i,:]).sum(), np.array(q_len[i]))

    mask = torch.einsum("bm,bn->bmn", (p_mask,q_mask)).float()

    p_result = torch.nn.functional.softmax(att * mask + mask_value * (1-mask), dim=-1)
    q_result = torch.nn.functional.softmax(att * mask + mask_value * (1 - mask), dim=-2)

    return p_result, q_result
class MatchNet(nn.Module):
    def __init__(self, mem_dim):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(4*mem_dim, mem_dim)
        self.map_linear_2 = nn.Linear(4 * mem_dim, mem_dim)
        self.sim_linear = nn.Linear(300,300, bias=False)
        self.bilstm = MaskLSTM(300,150)

    def forward(self, inputs, mask_value=0, topK=0):

        C_s, C_len, C_s_len, R_s, R_h, R_len = inputs
        att_weights = torch.einsum("bsik, bjk->bsij", (C_s, R_s))

        p_att, q_att = masked_softmax(att_weights, C_s_len, C_s.size(-2), R_len, R_s.size(-2))
        #store att matrix for checking
        check = q_att
        C_d = torch.einsum("bsij, bsik->bsjk", (q_att, C_s))
        C_d = torch.einsum("bsjk->bjsk", C_d)
        C_d = C_d.reshape(C_d.size(0)*C_d.size(1), C_d.size(2),-1)
        C_d = self.bilstm([C_d, C_len.unsqueeze(1).repeat(1,4).view(-1)]) #(batch*hint, step, 300)
        C_d = C_d.view(C_len.size(0), R_s.size(1), C_d.size(1), -1) #(b,h,s,d)


        att_weights_2 = torch.einsum("bjsk, bjk->bjs",(C_d,R_h ))
        p_att, q_att = masked_softmax_2(att_weights_2, R_len, R_s.size(-2), C_len, C_s.size(-3))
        C_h = torch.einsum("bjs, bjsk->bjk",(p_att, C_d))


        #compute matching score
        dist = C_h - R_h #(batch,hint,emb)
        dist = self.sim_linear(dist)
        dist = torch.norm(dist,dim=-1) #(batch, hint)
        dist = - dist.sum(dim=1)

        #compute ordering score
        step_att = torch.einsum("bjs->bsj",att_weights_2)
        _, max_index = torch.max(step_att, dim=1) #(batch, hint)
        step_max = step_att.gather(1, max_index.unsqueeze(1)).squeeze(1) #(batch, hint)
        all_diff = []
        for i in range(step_att.size(2) - 1):
            diff = step_max[:,i].unsqueeze(1).unsqueeze(2) - step_att[:,:,i+1:]
            mask = torch.arange(step_att.size(1)).unsqueeze(0).unsqueeze(2).expand_as(
                diff).cuda() \
                   < max_index[:,i].unsqueeze(1).unsqueeze(1).repeat(1,1,diff.size(2))
            diff = diff * mask.float()
            if not topK:
                max_diff,_ = diff.min(1)[0].min(1,keepdim=True)
            all_diff.append(max_diff)
        all_diff = torch.cat(all_diff, dim=1)
        all_diff = all_diff.min(dim=1)[0]

        return dist, all_diff, check
class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.3):
        super(MaskLSTM, self).__init__()
       
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional,
                                   dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs, return_y=False):
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i,:seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input*mask_in)

        H, (hn, cn) = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i,:seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask
        if return_y:
            #return only the final hidden state
            return output[:,-1]
        #return all the hidden states
        return output


class OrdMatch(nn.Module):
    def __init__(self,corpus, args):
        super(OrdMatch,self).__init__()
        self.emb_dim = 300
        self.hid_dim = args.hid_dim
        self.dropoutP = args.dropoutP
        self.use_cuda = args.cuda
        self.idx2word = corpus.dictionary.idx2word
        self.embs = nn.Embedding(len(corpus.dictionary), self.emb_dim)
        self.embs.weight.data.copy_(corpus.dictionary.embs)
        self.embs.weight.requires_grad = False

        self.encoder = MaskLSTM(self.emb_dim, self.hid_dim, dropoutP=self.dropoutP)
        self.match_module_text = MatchNet(self.hid_dim * 2)
        # self.match_module_img = MatchNet(self.hid_dim*2, self.dropoutP)
        # self.img_mapping = nn.Linear(2048, self.emb_dim)
        self.rank_module = nn.Linear(self.emb_dim, 1)
        self.hint_mapping = nn.Linear(self.emb_dim, self.emb_dim)

        self.drop_module = nn.Dropout(self.dropoutP)
        self.score_linear = nn.Linear(3,1)


    def forward(self, inputs):
        docs, qas, labels = inputs

        d_word, d_len, d_s_len = docs
        qa_word, qa_len = qas

        if self.use_cuda:
            d_word, d_len, d_s_len, qa_word, qa_len = d_word.cuda(), d_len.cuda(), d_s_len.cuda(), qa_word.cuda(), qa_len.cuda()

        # document: d_embs.shape = (batch_size, sent_num, sent_len, emb_dim)
        d_embs = self.drop_module(Variable(self.embs(d_word), requires_grad=True))
        # qa_pair: qa_embs.shape = (batch_size, option_num, hint_num, sent_len, emb_dim)
        qa_embs = self.drop_module(Variable(self.embs(qa_word), requires_grad=True))
        qa_mapped = nn.functional.relu(self.hint_mapping(qa_embs))
        qa_mapped = qa_mapped.mean(dim=3) #(batch ,option, hint, emb)

        # d_hidden.shape = (batch_size*sent_num, sent_len, emb_dim)
        d_hidden = self.encoder([d_embs.view(d_embs.size(0)*d_embs.size(1), d_embs.size(2), self.emb_dim), d_s_len.view(-1)])
        d_hidden = d_hidden.view(d_embs.size(0), d_embs.size(1), d_embs.size(2), d_embs.size(3))
        # qa_hidden.shape = (batch_zie*options_num, sent_len, emb_dim)
        qa_len = torch.ones(qa_len.size(0),qa_len.size(1)).long().cuda()
        qa_len = qa_len * 4
        qa_hidden = self.encoder([qa_mapped.view(qa_embs.size(0)*qa_embs.size(1), qa_embs.size(2), self.emb_dim), qa_len.view(-1)])
        qa_hidden = qa_hidden.view(qa_mapped.size(0), qa_mapped.size(1),qa_mapped.size(2),self.emb_dim)

        score1_match = []
        score2_match = []
        att = []
        for qa, qa_m, qa_len_s in zip(torch.unbind(qa_mapped,dim=1), torch.unbind(qa_hidden,dim=1), torch.unbind(qa_len,dim=1)):
            # d_hidden:(batch*sent_num, sent_len, dim), qa:(batch, sent_len, dim)
            score1, score2, att_s = self.match_module_text([d_hidden, d_len, d_s_len, qa, qa_m, qa_len_s])
            score1_match.append(score1)
            score2_match.append(score2)
            att.append(att_s)


        score1_match = torch.stack(score1_match, dim=1)
        score2_match = torch.stack(score2_match, dim=1)
        att = torch.stack(att, dim=1)


        output1 = torch.nn.functional.log_softmax(score1_match)
        output2 = torch.nn.functional.log_softmax(score2_match)

        return output1,output2, att
