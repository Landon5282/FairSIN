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
class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden-1, 2)
        
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return h

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

    discriminator_ = MLP_discriminator(args).to(args.device)
    optimizer_d_ = torch.optim.Adam([
        dict(params=discriminator_.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)
    
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
    
    # if args.dataset == 'credit':
    #     label_idx_0 = np.where(data.x[:, args.sens_idx].cpu() == 0)[0]
    #     label_idx_1 = np.where(data.x[:, args.sens_idx].cpu() == 1)[0]
    #     min_label_number = min(len(label_idx_0), len(label_idx_1))
    #     idx_train_0 = label_idx_0[:int(0.5 * min_label_number)]
    #     idx_train_1 = label_idx_1[:int(0.5 * min_label_number)]
    #     idx_train = np.concatenate([idx_train_0, idx_train_1])
    #     idx_val_0 = label_idx_0[int(0.5 * min_label_number):int(0.75 * min_label_number)]
    #     idx_val_1 = label_idx_1[int(0.5 * min_label_number):int(0.75 * min_label_number)]
    #     idx_val = np.concatenate([idx_val_0, idx_val_1])
    #     idx_test_0 = label_idx_0[int(0.75 * min_label_number):]
    #     idx_test_1 = label_idx_1[int(0.75 * min_label_number):]
    #     idx_test = np.concatenate([idx_test_0, idx_test_1])
    
    #     data.train_mask = index_to_mask(data.x.shape[0], torch.LongTensor(idx_train))
    #     data.val_mask = index_to_mask(data.x.shape[0], torch.LongTensor(idx_val))
    #     data.test_mask = index_to_mask(data.x.shape[0], torch.LongTensor(idx_test))

    # 检查文件是否存在
    if os.path.isfile(args.dataset+'_hadj.pt'):
        print('########## sample already done #############')
        # 加载数据
        new_adj = torch.load(args.dataset+'_hadj.pt')
    else:
        # pretrain neighbor predictor
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        # neighbor select
        print('sample begin')
        # 计算异质邻居
        new_adj = torch.zeros((data.adj.shape[0], data.adj.shape[0])).int()
        # 遍历每一个节点
        for i in tqdm(range(data.adj.shape[0])):
            # 获取节点i的所有邻居节点索引
            neighbor = torch.tensor(data.adj[i].nonzero()).to(args.device)
            # 筛选出与节点i所属类别不同的邻居节点
            mask = (data.sens[neighbor[1]] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            # 将节点i和其所有邻居节点之间的边加入邻接矩阵
            new_adj[i, h_nei_idx] = 1
        print('select done')
        # 保存数据
        torch.save(new_adj, args.dataset+'_hadj.pt')
    
    c_X = data.x
    new_adj = new_adj.cpu()
    # 计算度矩阵和异质邻居特征
    deg = np.sum(new_adj.numpy(), axis=1)
    deg = torch.from_numpy(deg).cpu()
    indices = torch.nonzero(new_adj)
    values = new_adj[indices[:, 0], indices[:, 1]]
    mat = torch.sparse_coo_tensor(indices.t(), values, new_adj.shape).float().cpu()
    h_X = torch.spmm(mat,(data.x).cpu()) / deg.unsqueeze(-1) 
    # 检查每一行是否包含 NaN 值
    mask = torch.any(torch.isnan(h_X), dim=1)
    # 删除包含 NaN 值的行
    h_X = h_X[~mask].to(args.device)
    c_X = c_X[~mask].to(args.device)
    print('node avg degree:',data.edge_index.shape[1]/data.adj.shape[0],' heteroneighbor degree mean:',deg.float().mean(),' node without heteroneghbor:',(deg == 0).sum())
    deg_norm = deg
    deg_norm[deg_norm == 0] = 1
    deg_norm = deg_norm.to(args.device)
    
    model = MLP(len(data.x[0]),args.hidden,len(data.x[0])).to(args.device)
    optimizer = torch.optim.Adam([
            dict(params=model.parameters(), weight_decay=0)], lr=args.m_lr)


    from sklearn.model_selection import train_test_split

    indices = np.arange(c_X.shape[0]) # help for check the index after split
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]
    
    test_acc1, test_acc2, test_acc3, test_H_score1, test_H_score2, test_H_score3 = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)    
    for count in pbar:
        seed_everything(count+args.seed)
        model.reset_parameters()
        discriminator.reset_parameters()
        discriminator_.reset_parameters()
        
        best_val_tradeoff = 0
        best_val_loss = math.inf
                

        # train mlp
        for m_epoch in range(0, args.m_epoch):                
            model.train() # prep model for training
            
            output = model(X_train)
            train_loss = torch.nn.functional.mse_loss(output, y_train)
            # backward pass: compute gradient of the loss with respect to model parameters
            train_loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            model.eval()
            output = model(X_test)
            valid_loss = torch.nn.functional.mse_loss(output, y_test)
            

            # print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(
            #     m_epoch, 
            #     train_loss,
            #     valid_loss
            #     ))
                
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = model.state_dict()
                # print('best_val_loss:',m_epoch)

        model.load_state_dict(best_mlp_state)

        # raw feature 
        x = data.x
        x = x[:, [i for i in range(data.x.size(1)) if i != args.sens_idx]]
        best_val_loss = math.inf
        # train discriminator to recognize the sensitive group
        encoder.train()
        for epoch_d in range(0, args.d_epochs):
            optimizer_d.zero_grad()
            discriminator.train()
            output = discriminator(x)
            loss_d = F.cross_entropy(output[data.train_mask], data.x[:, args.sens_idx][data.train_mask].long())
            # loss_d = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,1] - output[data.test_mask][:,0])).sum().item() / data.test_mask.sum().item()
            loss_d.backward()
            optimizer_d.step()

            discriminator.eval()
            output = discriminator(x)
            valid_loss = F.cross_entropy(output[data.val_mask], data.x[:, args.sens_idx][data.val_mask].long())
            
            # print(f'epoch:{epoch_d},loss_d:{loss_d},valid loss:{valid_loss}')
            if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = discriminator.state_dict()
                    # print('best_val_loss:',epoch_d)
        
        discriminator.load_state_dict(best_mlp_state)
        discriminator.eval()
        output = discriminator(x)
        pred_val1 = (output[data.test_mask][:,1] > output[data.test_mask][:,0]).type_as(data.y) # 将output转换为预测值
        acc1 = pred_val1.eq(data.x[:, args.sens_idx][data.test_mask]).sum().item() / data.test_mask.sum().item()

        probs = torch.nn.functional.softmax(output, dim=1)
        prob1 = probs[:, 0][data.test_mask].mean()
        prob_all1 = probs[:, 0].mean()
        H_score1 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1])).sum().item() / data.test_mask.sum().item()
        pred_val1 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1]))


        ### message passing
        best_val_loss = math.inf
        # print('-----------after message passing:')
        discriminator_.reset_parameters()
        discriminator_.train()
        encoder.eval()
        emb = torch.spmm(data.adj_norm_sp, data.x)  
        emb = emb[:, [i for i in range(0,data.x.size(1)) if i != args.sens_idx]]
        for epoch_d in range(0, args.d_epochs):
            discriminator_.train()
            optimizer_d_.zero_grad()
            output = discriminator_(emb)
            loss_d_ =F.cross_entropy(output[data.train_mask], data.x[:, args.sens_idx][data.train_mask].long())
            loss_d_.backward()
            optimizer_d_.step()

            discriminator_.eval()
            output = discriminator_(emb)
            valid_loss = F.cross_entropy(output[data.val_mask], data.x[:, args.sens_idx][data.val_mask].long())
            
            # print(f'epoch:{epoch_d},loss_d:{loss_d_},valid loss:{valid_loss}')
            if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = discriminator_.state_dict()
                    # print('best_val_loss:',epoch_d)

        discriminator_.load_state_dict(best_mlp_state)
        discriminator_.eval()
        output = discriminator_(emb)
        pred_val2 = (output[data.test_mask][:,1] > output[data.test_mask][:,0]).type_as(data.y) # 将output转换为预测值
        acc2 = pred_val2.eq(
            data.x[:, args.sens_idx][data.test_mask]).sum().item() / data.test_mask.sum().item()
        H_score2 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1])).sum().item() / data.test_mask.sum().item()
        pred_val2 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1]))
        probs = torch.nn.functional.softmax(output, dim=1)
        prob2 = probs[:, 0][data.test_mask].mean()
        prob_all2 = probs[:, 0].mean()

        ### neturalization
        best_val_loss = math.inf
        discriminator_.reset_parameters()
        origin_state = discriminator_.state_dict()
        encoder.eval()
        model.eval()
        emb = data.x + args.delta * model(data.x).detach()
        # emb = torch.spmm(data.adj_norm_sp, emb)  
        emb = emb[:, [i for i in range(0,data.x.size(1)) if i != args.sens_idx]]
        for epoch_d in range(0, args.d_epochs):
            discriminator_.train()
            optimizer_d_.zero_grad()
            output = discriminator_(emb)
            loss_d_ =F.cross_entropy(output[data.train_mask], data.x[:, args.sens_idx][data.train_mask].long())
            loss_d_.backward()
            optimizer_d_.step()

            discriminator_.eval()
            output = discriminator_(emb)
            valid_loss = F.cross_entropy(output[data.val_mask], data.x[:, args.sens_idx][data.val_mask].long())
            
            # print(f'epoch:{epoch_d},loss_d:{loss_d_},valid loss:{valid_loss}')
            if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = discriminator_.state_dict()
                    # print('best_val_loss:',epoch_d)

        discriminator_.load_state_dict(best_mlp_state)
        discriminator_.eval()
        output = discriminator_(emb)
        pred_val3 = (output[data.test_mask][:,1] > output[data.test_mask][:,0]).type_as(data.y) # 将output转换为预测值
        acc3 = pred_val3.eq(
            data.x[:, args.sens_idx][data.test_mask]).sum().item() / data.test_mask.sum().item()
        H_score3 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1])).sum().item() / data.test_mask.sum().item()
        pred_val3 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1]))
  

        ### neturalization + mp
        best_val_loss = math.inf
        discriminator_.load_state_dict(origin_state)
        encoder.eval()
        model.eval()
        emb = data.x + args.delta * model(data.x).detach()
        emb = torch.spmm(data.adj_norm_sp, emb) / 2  
        emb = emb[:, [i for i in range(0,data.x.size(1)) if i != args.sens_idx]]
        for epoch_d in range(0, args.d_epochs):
            discriminator_.train()
            optimizer_d_.zero_grad()
            output = discriminator_(emb)
            loss_d_ =F.cross_entropy(output[data.train_mask], data.x[:, args.sens_idx][data.train_mask].long())
            loss_d_.backward()
            optimizer_d_.step()

            discriminator_.eval()
            output = discriminator_(emb)
            valid_loss = F.cross_entropy(output[data.val_mask], data.x[:, args.sens_idx][data.val_mask].long())
            
            # print(f'epoch:{epoch_d},loss_d:{loss_d_},valid loss:{valid_loss}')
            if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = discriminator_.state_dict()
                    # print('best_val_loss:',epoch_d)

        discriminator_.load_state_dict(best_mlp_state)
        discriminator_.eval()
        output = discriminator_(emb)
        pred_val4 = (output[data.test_mask][:,1] > output[data.test_mask][:,0]).type_as(data.y) # 将output转换为预测值
        acc4 = pred_val4.eq(
            data.x[:, args.sens_idx][data.test_mask]).sum().item() / data.test_mask.sum().item()
        H_score4 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1])).sum().item() / data.test_mask.sum().item()
        pred_val4 = (1/2 + 1/2 * F.tanh(output[data.test_mask][:,0] - output[data.test_mask][:,1]))

        

        print('======' + args.dataset + '======')
        # print('acc1:',acc1)
        # print('acc2:',acc2)
        # print('acc3:',acc3)
        print('score1:',H_score1)
        print('score2:',H_score2)
        print('score3:',H_score3)
        print('score4:',H_score4)

    # print('auc_roc:', np.mean(auc_roc[:(count + 1)]))
    # print('f1:', np.mean(f1[:(count + 1)]))
    # print('acc:', np.mean(acc[:(count + 1)]))
    # print('Statistical parity:', np.mean(parity[:(count + 1)]))
    # print('Equal Opportunity:', np.mean(equality[:(count + 1)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--d_wd', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0.0001)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0.0001)
    parser.add_argument('--prop', type=str, default='scatter')       
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--m_epoch', type=int, default=100)
    parser.add_argument('--m_lr', type=float, default=0.01)

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

    run(data, args)
    # print('======' + args.dataset+'_avg' +'======')
    # print('acc1:', np.mean(acc1) ,'±',np.std(acc1), sep='')
    # print('acc2:', np.mean(acc2), '±' ,np.std(acc2), sep='')
    # print('acc3:', np.mean(acc3), '±' ,np.std(acc3), sep='')
    # print('H_score1:', np.mean(H_score1), '±' ,np.std(H_score1), sep='')
    # print('H_score2:', np.mean(H_score2), '±', np.std(H_score2), sep='')
    # print('H_score3:', np.mean(H_score3), '±', np.std(H_score3), sep='')


