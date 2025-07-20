import torch
import torch.nn as nn
from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.utils.run_util import check_device
from torch.nn import Sequential as Seq, Linear as Lin
import logging
import torch.optim as optim
from torch_scatter import scatter
from torch.linalg import vector_norm

model_name = "SWE_GNN_V1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SWE_GNN_ModelWrapper")

def make_mlp(input_dim, output_dim, hidden_dim, n_layers=2, bias=False):
    device = check_device()
    layers = []
    if n_layers == 1:
        layers.append(Lin(input_dim, hidden_dim, bias=bias, device=device))
        layers = layers + add_norm_droput_activation(hidden_dim, device)
    else: 
        layers.append(Lin(input_dim, hidden_dim, bias=bias, device=device))
        
        for _ in range(n_layers - 2):
            layers.append(Lin(hidden_dim, hidden_dim, bias=bias, device=device))
            layers = layers + add_norm_droput_activation(hidden_dim, device)
        layers.append(Lin(hidden_dim, output_dim, bias=bias, device=device))
        layers = layers + add_norm_droput_activation(output_dim, device)
    return Seq(*layers)

def add_norm_droput_activation(hidden_dim, device):
    layers= []
    # layers.append(nn.Dropout(0))
    layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, device=device))
    layers.append(nn.PReLU())
    return layers   
        
        

class BaseFloodModel(nn.Module):
    '''Base class for modelling flood inundation
    ------
    previous_t: int
        dataset-specific parameter that indicates the number of previous times steps given as input
    seed: int
        seed used for replicability
    '''
    def __init__(self, previous_t=1, seed=42, with_WL=True, device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        self.previous_t = previous_t
        self.with_WL = with_WL
        self.device = device
        self.out_dim = 2
            
    def _mask_small_WD(self, x, epsilon=0.001):        
        x[:,0][x[:,0].abs() < epsilon] = 0

        # Mask velocities where there is no water
        x[:,1:][x[:,0] == 0] = 0

        return x
    
class GNN(BaseFloodModel):
    def __init__(self, node_features, edge_features, hid_features=32, K=2, 
                 n_GNN_layers=2, mlp_layers=1, with_WL=False, 
                 normalise=True, with_filter_matrix=False, with_gradient=False, **base_model_kwargs):
        super(GNN, self).__init__(**base_model_kwargs)
        self.hid_features = hid_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.with_WL = with_WL
        self.dynamic_node_features = self.previous_t * self.out_dim
        self.static_node_features = node_features - self.dynamic_node_features + self.with_WL
        
        # Edge encoder
        self.edge_features = hid_features
        self.edge_encoder = make_mlp(edge_features, hid_features, hid_features, n_layers=2, bias=True)
        
        ## Node encoder
        self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, hid_features, hid_features, n_layers=2)
        self.static_node_encoder = make_mlp(self.static_node_features, hid_features, hid_features, n_layers=2, bias=True)
        
        #GNN
        convs = nn.ModuleList()
        for _ in range(n_GNN_layers):
            convs.append(SWEGNN(hid_features, hid_features, 
                                self.edge_features, K=K, normalize=normalise, 
                                with_filter_matrix=with_filter_matrix, with_gradient=with_gradient, mlp_layers=mlp_layers))
        self.gnn_procesor = convs
        self.gnn_activations = Seq(*[nn.Tanh()] * n_GNN_layers)
        
        # Decoder 
        self.node_decoder = make_mlp(hid_features, self.out_dim, hid_features, n_layers=2)
        
    
    def forward(self, graph):
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        
        #Build encoder and decoder block
        # 1. Node and edge encoder 
        edge_attr = self.edge_encoder(edge_attr)
        x0 = x
        x_s = x[:, :self.static_node_features-self.with_WL]
        x_t = x[:, self.static_node_features-self.with_WL:]
        
        if self.with_WL:
            #Add water level to static node features
            WL = x_s[:, -1] + x_t[:, -self.out_dim]
            x_s = torch.cat((x_s, WL.unsqueeze(-1)), 1)

        x_s = self.static_node_encoder(x_s)
        x = x_t = self.dynamic_node_encoder(x_t)
        
        #2. Processor
        for i, conv in enumerate(self.gnn_procesor):
            x = conv(x_s, x_t, edge_index, edge_attr)
        
            #Add activation
            x = self.gnn_activations[i](x)
            
            x_t = x
        
        #3. Decoder
        x = self.node_decoder(x_t)
        
        #Add residual connection
        x = x + x0[:, -self.out_dim:]
        
        # Mask small water depth and velocities
        x = self._mask_small_WD(x, epsilon=0.3)
        
        return self.edge_mlp(x)

class SWEGNN(nn.Module):
    def __init__(self, static_node_features:int, dynamic_node_features:int, edge_features:int, K:int, normalize=True,
                 with_filter_matrix=True, with_gradient=True, **mlp_kwargs):
        super(SWEGNN, self).__init__()
        self.edge_features = edge_features
        self.edge_input_size = edge_features + static_node_features * 2 + dynamic_node_features * 2
        self.edge_ouput_size = dynamic_node_features
        self.with_filter_matrix = with_filter_matrix
        self.with_gradient = with_gradient
        hidden_size = self.edge_input_size * 2
        self.K = K
        self.normalize = normalize
        with_gradient = with_gradient
        self.device = check_device()
        self.edge_mlp = make_mlp(self.edge_input_size, self.edge_ouput_size, hidden_size, **mlp_kwargs)
        
        
        self.filter_matrix = torch.nn.ModuleList([
            nn.Linear(dynamic_node_features, dynamic_node_features, bias=False) for _ in range(K+1)
        ])
            
    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:  
        """
        Forward pass of the SWE GNN model.
        
        :param x_s: Static node features (shape: [num_nodes, static_node_features])
        :param x_t: Dynamic node features (shape: [num_nodes, dynamic_node_features])
        :param edge_index: Edge indices (shape: [2, num_edges])
        :param edge_attr: Edge attributes (shape: [num_edges, edge_features])
        :return: Updated dynamic node features (shape: [num_nodes, dynamic_node_features])
        """
        row, col = edge_index
        num_nodes = x_t.size(0)
        out = self.filter_matrix[0].forward(x_t.clone())
  
        for k in range(1, self.K):
            mask = out.sum(1) != 0
            mask_row = row[mask]
            mask_col = col[mask]
            edge_index_mask = mask_row + mask_col
            
            # Edge update
            e_ij = torch.cat([x_s[row][edge_index_mask], x_s[col][edge_index_mask],out[row][edge_index_mask], 
                              out[col][edge_index_mask], edge_attr[edge_index_mask]], dim=1)
            w_ij = self.edge_mlp(e_ij)
            
            #Normalize
            w_ij = w_ij /vector_norm(w_ij, dim=1, keepdim=True)
            w_ij.masked_fill_(torch.isnan(w_ij), 0)
                
            #Node update
            shift_sum = (out[col][edge_index_mask] - out[row][edge_index_mask]) * w_ij
            scattered = scatter(shift_sum, col[edge_index_mask], reduce='sum', dim=0, dim_size=num_nodes)
            out = out + self.filter_matrix[k+1].forward(scattered)
        return out

    
class SWEGNNModelWrapper(ModelWrapper):
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        self.device = check_device()
        self.validation_event = self.config.fold  + 1
        self.tuninig_mode = config.args.get('tuning_mode', False)
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = GNNDataManager()
        
    def init_model(self) -> bool:
        try:
            self.create_dataset()
            self.model = GNN().to(self.device)
            self.loss_fn= nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            return True
        
        except Exception as e:
            logger.error(f"Error creating 1DCNN model: {e}")
            return False
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        return super().test_model()
    
