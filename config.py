import argparse


parser = argparse.ArgumentParser(description='RecipeQA')
parser.add_argument('--task', type=str, default='TC')
parser.add_argument('--emb_dim', type=int, default=300, help='size of word embeddings')
parser.add_argument('--hid_dim', type=int, default=150, help='hidden size')
parser.add_argument('--lr',type=float,default=1e-3,help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='epoch')
parser.add_argument('--batch_size', type=int, default=5, metavar='N', help='batch size')
parser.add_argument('--dropoutP', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', type=bool, default=True, help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='use which GPU device')
parser.add_argument('--interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--exp_idx', type=str,  default='1', help='experiment index')
parser.add_argument('--lamda', type=float, default='0.01',help='lambda factor for interpolated loss')
parser.add_argument('--save_path',type=str, default='./outputs')
parser.add_argument('--data_path',type=str, default='./data')




args = parser.parse_args()

