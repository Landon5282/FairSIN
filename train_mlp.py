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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from torch.optim.lr_scheduler import ExponentialLR

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
    
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
        new_adj = torch.zeros((data.adj.shape[0], data.adj.shape[0])).int()
        for i in tqdm(range(data.adj.shape[0])):
            neighbor = torch.tensor(data.adj[i].nonzero()).to(args.device)
            mask = (data.sens[neighbor[1]] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            new_adj[i, h_nei_idx] = 1
        print('select done')
        torch.save(new_adj, args.dataset+'_hadj.pt')
    
    c_X = data.x
    new_adj = new_adj.cpu()

    adj = torch.from_numpy(data.adj.toarray()).cpu()
    adj_new = adj -  new_adj

    num_homo_neighbors = adj_new.sum(dim=1)
    num_hetero_neighbors = new_adj.sum(dim=1)
    diffs = num_homo_neighbors - num_hetero_neighbors + 1

    lam = diffs.reshape(-1, 1).to(args.device, dtype=torch.float)

    deg = np.sum(new_adj.numpy(), axis=1)
    deg = torch.from_numpy(deg).cpu()
    indices = torch.nonzero(new_adj)
    values = new_adj[indices[:, 0], indices[:, 1]]
    mat = torch.sparse_coo_tensor(indices.t(), values, new_adj.shape).float().cpu()
    h_X = torch.spmm(mat,(data.x).cpu()) / deg.unsqueeze(-1) 

    mask = torch.any(torch.isnan(h_X), dim=1)

    h_X = h_X[~mask].to(args.device)
    c_X = c_X[~mask].to(args.device)
    print('node avg degree:',data.edge_index.shape[1]/data.adj.shape[0],' heteroneighbor degree mean:',deg.float().mean(),' node without heteroneghbor:',(deg == 0).sum())
    deg_norm = deg
    deg_norm[deg_norm == 0] = 1
    deg_norm = deg_norm.to(args.device)
    
    model = MLP(len(data.x[0]),args.hidden,len(data.x[0])).to(args.device)
    optimizer = torch.optim.Adam([
            dict(params=model.parameters(), weight_decay=0.005)], lr=0.1)


    from sklearn.model_selection import train_test_split

    indices = np.arange(c_X.shape[0]) # help for check the index after split
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]

    for count in pbar:
        seed_everything(count + args.seed)
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()
        model.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
        
            
        for epoch in range(0, args.epochs):
            # train mlp
            for m_epoch in range(0, args.m_epoch):                
                model.train() # prep model for training
                optimizer.zero_grad()
                
                output = model(X_train)
                train_loss = torch.nn.functional.mse_loss(output, y_train)
                # backward pass: compute gradient of the loss with respect to model parameters
                train_loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                
                model.eval()
                output = model(X_test)
                valid_loss = torch.nn.functional.mse_loss(output, y_test)

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = model.state_dict()
                # if m_epoch % 5 == 0:
                #     print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(
                #         m_epoch, 
                #         train_loss,
                #         valid_loss
                #         ))
            model.load_state_dict(best_mlp_state)
            model.eval()
            

            # train classifier
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()
                optimizer.zero_grad()
                h = encoder(data.x + lam * args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                # h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                output = classifier(h)
                loss_c = F.binary_cross_entropy_with_logits(
                    output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()
                # optimizer.step()

            if args.d == 'yes':
                # train discriminator to recognize the sensitive group
                discriminator.train()
                encoder.train()
                for epoch_d in range(0, args.d_epochs):
                    optimizer_d.zero_grad()
                    optimizer_e.zero_grad()
                    optimizer.zero_grad()

                    h = encoder((data.x + lam * args.delta * model(data.x)) , data.edge_index, data.adj_norm_sp)
                    # h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                    output = discriminator(h)

                    loss_d = criterion(output.view(-1),
                                        data.x[:, args.sens_idx])

                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_e.step()

            # evaluate classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(
                data.x, classifier, discriminator, encoder, data, args)
            # print('*****************')
            # print(epoch, 'Acc:', accs['val'], 'AUC_ROC:', auc_rocs['val'], 'F1:', F1s['val'],
            #       'Parity:', tmp_parity['val'], 'Equality:', tmp_equality['val'],'tradeoff:',auc_rocs['val']+F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']))
            print(epoch, 'Acc:', accs['test'], 'F1:', F1s['test'],
                  'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'],'tradeoff:',auc_rocs['test']+F1s['test'] + accs['test'] - args.alpha * (tmp_parity['test'] + tmp_equality['test']))

            # if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
            #     test_acc = accs['test']
            #     test_auc_roc = auc_rocs['test']
            #     test_f1 = F1s['test']
            #     test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

            #     best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
            #         accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
                
            if  auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                print('best_val_tradeoff',epoch)
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
                
    
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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--d_lr', type=float, default=0.1)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.1)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.1)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--prop', type=str, default='scatter')       
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=5)
    parser.add_argument('--m_epoch', type=int, default=10)
    parser.add_argument('--d', type=str, default='no')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 2-1 # binary classes are 0,1 

    acc, f1, auc_roc, parity, equality = run(data, args)
    print('======' + args.dataset + args.encoder + '======')
    print('Acc:', round(np.mean(acc) * 100,2), '±' ,round(np.std(acc) * 100,2), sep='')
    print('f1:', round(np.mean(f1) * 100,2), '±' ,round(np.std(f1) * 100,2), sep='')
    print('parity:', round(np.mean(parity) * 100,2), '±', round(np.std(parity) * 100,2), sep='')
    print('equality:', round(np.mean(equality) * 100,2), '±', round(np.std(equality) * 100,2), sep='')
