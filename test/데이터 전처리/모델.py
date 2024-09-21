import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import warnings
warnings.filterwarnings('ignore')

#      복잡 모델
def GCN_Model(data):
    class GCN(torch.nn.Module):
        def __init__(self, num_node_features, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_node_features, 16)
            self.bn1 = BatchNorm1d(16)
            self.conv2 = GCNConv(16, 32)
            self.bn2 = BatchNorm1d(32)
            self.conv3 = GCNConv(32, 64)
            self.bn3 = BatchNorm1d(64)
            self.conv4 = GCNConv(64, num_classes)
            # self.bn4 = BatchNorm1d(128)
            # self.conv5 = GCNConv(128, num_classes)
            # self.bn5 = BatchNorm1d(256)
            # self.conv6 = GCNConv(256, num_classes)
            # self.bn6 = BatchNorm1d(512)
            # self.conv7 = GCNConv(512, num_classes)
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.gelu(x)
            # x = self.dropout(x)

            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.gelu(x)
            # x = self.dropout(x)

            x = self.conv3(x, edge_index)
            x = self.bn3(x)
            x = F.gelu(x)
            # x = self.dropout(x)
            
            x = self.conv4(x, edge_index)
            # x = self.bn4(x)
            # x = F.gelu(x)
            # x = self.dropout(x)
            
            # x = self.conv5(x, edge_index)
            # x = self.bn5(x)
            # x = F.gelu(x)
            # x = self.dropout(x)
            
            # x = self.conv6(x, edge_index)
            # x = self.bn6(x)
            # x = F.gelu(x)
            # x = self.dropout(x)
            
            # x = self.conv7(x, edge_index)

            return F.log_softmax(x, dim=1)


    new_data = data

    # 노드 특징 행렬 생성
    x_new = torch.tensor(new_data.values, dtype=torch.float)

    model = GCN(num_node_features=x_new.size(1), num_classes=3)  # 피처 개수와 클래스 개수에 맞춰 초기화

    # 모델 로드
    model.load_state_dict(torch.load('C:/big18/final/test/모델/gcn1.pth'))
    model.eval()

    edge_index = []
    num_rows = len(new_data)
    # for j in range(1, num_rows):
    #     edge_index.append([j, j - 1])
    
    for i in range(num_rows - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])


    edge_index_new = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 그래프 데이터 객체 생성
    new_graph_data = Data(x=x_new, edge_index=edge_index_new)

    # 예측 수행
    with torch.no_grad():
        output = model(new_graph_data)
        pred = output.max(dim=1)[1].cpu().numpy()

    print("예측 :", pred[-1])
    return pred[-1]
