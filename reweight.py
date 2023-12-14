from dataset import *
from model import *
from utils import *
from evaluation import *
import argparse
from tqdm import tqdm
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from torch.optim.lr_scheduler import ExponentialLR
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, h):
        h = self.lin1(h)
        h = self.lin2(h)
        return h

def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device) 
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)


    if os.path.isfile(args.dataset+'_hadj.pt'):
        print('########## sample already done #############')
        new_adj = torch.load(args.dataset+'_hadj.pt')
    else:
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        print('sample begin')
        for i in tqdm(range(data.adj.shape[0])):
            neighbor = torch.tensor(data.adj[i].nonzero()).to(args.device)
            mask = (data.sens[neighbor[1]] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            new_adj[i, h_nei_idx] = 1
        print('select done')
        torch.save(new_adj, args.dataset+'_hadj.pt')

    new_adj = new_adj.cpu()
    new_adj_sp = new_adj.numpy()
    new_adj_sp = sp.coo_matrix(new_adj)
    new_adj_sp = sp.csr_matrix((new_adj_sp.data, (new_adj_sp.row, new_adj_sp.col)), shape=data.adj.shape)
    data.adj = data.adj + sp.eye(data.adj.shape[0]) + args.delta * new_adj_sp
    adj_norm = sys_normalized_adjacency(data.adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    data.adj_norm_sp = adj_norm_sp

    deg = np.sum(new_adj.numpy(), axis=1)
    deg = torch.from_numpy(deg).cpu()
    print('node avg degree:',data.edge_index.shape[1]/data.adj.shape[0],' heteroneighbor degree mean:',deg.float().mean(),' node without heteroneghbor:',(deg == 0).sum())


    for count in pbar:
        seed_everything(count + args.seed)
        classifier.reset_parameters()
        encoder.reset_parameters()
        # model.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
        
        for epoch in range(0, args.epochs):
            
            # train classifier

            for epoch_c in range(0, args.c_epochs):
                classifier.train()
                encoder.train()
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()

                # h = encoder(data.x + model(data.x), data.edge_index, data.adj_norm_sp)
                h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                output = classifier(h)

                loss_c = F.binary_cross_entropy_with_logits(
                    output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()
                
            # evaluate classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(
                data.x, classifier, discriminator, encoder, data, args)

            print(epoch, 'Acc:', accs['test'], 'AUC_ROC:', auc_rocs['test'], 'F1:', F1s['test'],
                  'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'],'tradeoff:',auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']))

            # if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
            #     test_acc = accs['test']
            #     test_auc_roc = auc_rocs['test']
            #     test_f1 = F1s['test']
            #     test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

            #     best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
            #         accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
                
            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                print('best_val_tradeoff',epoch)
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                    accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

                # print('=====VALIDATION=====', epoch, epoch_g)
                # print('Utility:', auc_rocs['val'] + F1s['val'] + accs['val'],
                #       'Fairness:', tmp_parity['val'] + tmp_equality['val'])

                # print('=====VALIDATION-BEST=====', epoch, epoch_g)
                # print('Utility:', args.best_val_model_utility,
                #       'Fairness:', args.best_val_fair)

                # print('=====TEST=====', epoch)
                # print('Acc:', test_acc, 'AUC_ROC:', test_auc_roc, 'F1:', test_f1,
                #       'Parity:', test_parity, 'Equality:', test_equality)

                # print('=====epoch:{}====='.format(epoch))
                # print('sens_acc:', (((output.view(-1) > 0.5) & (data.x[:, args.sens_idx] == 1)).sum() + ((output.view(-1) < 0.5) &
                #                                                                                          (data.x[:, args.sens_idx] == 0)).sum()).item() / len(data.y))
        
        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality
        # print('auc_roc:', np.mean(auc_roc[:(count + 1)]))
        # print('f1:', np.mean(f1[:(count + 1)]))
        # print('acc:', np.mean(acc[:(count + 1)]))
        # print('Statistical parity:', np.mean(parity[:(count + 1)]))
        # print('Equal Opportunity:', np.mean(equality[:(count + 1)]))

    return acc, f1, auc_roc, parity, equality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--d_lr', type=float, default=0.002)
    parser.add_argument('--d_wd', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0.0001)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0.0001)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--prop', type=str, default='scatter')       
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--m_epoch', type=int, default=100)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.init()
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 2-1 # binary classes are 0,1 

    # print((data.y == 1).sum(), (data.y == 0).sum())
    # print((data.y[data.train_mask] == 1).sum(),
    #       (data.y[data.train_mask] == 0).sum())
    # print((data.y[data.val_mask] == 1).sum(),
    #       (data.y[data.val_mask] == 0).sum())
    # print((data.y[data.test_mask] == 1).sum(),
    #       (data.y[data.test_mask] == 0).sum())

    args.train_ratio, args.val_ratio = torch.tensor([
        (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), torch.tensor([
            (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(
        args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[
        data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

    # print(args.val_ratio, data.y[data.val_mask])

    acc, f1, auc_roc, parity, equality = run(data, args)
    print('======' + args.dataset + args.encoder + '======')
    print('auc_roc:', round(np.mean(auc_roc)* 100,2),'±',round(np.std(auc_roc) * 100,2), sep='')
    print('Acc:', round(np.mean(acc) * 100,2), '±' ,round(np.std(acc) * 100,2), sep='')
    print('f1:', round(np.mean(f1) * 100,2), '±' ,round(np.std(f1) * 100,2), sep='')
    print('parity:', round(np.mean(parity) * 100,2), '±', round(np.std(parity) * 100,2), sep='')
    print('equality:', round(np.mean(equality) * 100,2), '±', round(np.std(equality) * 100,2), sep='')