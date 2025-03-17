import argparse

parser = argparse.ArgumentParser(description='SCCGC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default="acm")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta', type=int, default=10)
parser.add_argument('--order', type=int, default=3, help='to compute order-th power of adj')
parser.add_argument('--influence', default=True, action='store_true', help='Use Inluence contrastive')

args = parser.parse_args()
print("Network setting…")

if args.name == 'acm':
    args.lr = 2e-3
    args.n_clusters = 3
    args.n_input = 1870
    args.gama = 0.01
    args.alpha = 0.1

elif args.name == 'dblp':
    args.lr = 2e-3
    args.n_clusters = 4
    args.n_input = 334
    args.alpha = 0.05
    args.gama = 0.01


elif args.name == 'pubmed':
    args.lr = 2e-3
    args.n_clusters = 3
    args.n_input = 500
    args.gama = 0.01
    args.alpha = 0.1


else:
    print("error!")
