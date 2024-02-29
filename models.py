import math
import pyvqnet.nn as nn
import pyvqnet.qnn as qnn
#import QLayer as qnn
import pyvqnet.tensor as tensor
from pyvqnet.nn.activation import *
from pyvqnet.tensor import QTensor as Tensor
import numpy as np
from pyvqnet.dtype import kfloat64, kfloat32
import qpandalite
import pyqpanda as pq
from pyvqnet.qnn.measure import expval
from pyvqnet.tensor import pad_sequence, pad_packed_sequence, pack_pad_sequence
from tqdm import tqdm


DUMMY = False
dtype = kfloat32
data_type = np.float32

khan_mapping = [
    {0:24,1:25,2:26,3:27,4:28},
    {0:0,1:1,2:2,3:3,4:4},
    {0:42,1:48,2:49,3:50,4:44},
    # {0:54,1:60,2:66,3:67,4:61},
    # {0:4,1:3,2:9,3:10,4:11},
    # {0:42,1:48,2:49,3:50,4:44},
    # {0:26,1:32,2:38,3:39,4:40},
    # {0:8,1:14,2:15,3:16,4:17},
    # {0:35,1:41,2:47,3:46,4:52},
    # {0:13,1:12,2:18,3:24,4:30},
    # {0:57,1:56,2:62,3:68,4:69}
]

def vqc_encoding(qubits, inputs):
    """
    :param qubits: QVec[n_features]
    :param inputs: Array[n_features]
    :return: circuit
    """
    circuit = pq.QCircuit()

    ###处理科学计数法的数字
    ry_params = np.arctan(inputs)
    rz_params = np.arctan(inputs ** 2)
    for i, (ry_param, rz_param) in enumerate(zip(ry_params, rz_params)):
        circuit << pq.H(qubits[i])
        circuit << pq.RY(qubits[i], ry_param)
        circuit << pq.RZ(qubits[i], rz_param)
    return circuit


def vqc_ansatz(qubits, params):
    """
    :param qubits: QVec[n_features]
    :param params: Array[3, n_features]
    :return: circuit
    """
    circuit = pq.QCircuit()
    n_qubits = len(qubits)

    """
    加载CNOT门更改线型拓扑后重新训练的模型的话，上述部分要改为下述部分
    # Entangling layer
    for j in range(n_qubits-1):
        circuit << CNOT_LITE(qubits[j], qubits[(j + 1)])
    """
    for j in range(n_qubits-1):
        circuit << pq.CNOT(qubits[j], qubits[(j + 1)])

    # Variational layer
    for i in range(n_qubits):
        circuit << pq.RX(qubits[i], params[0][i])
        circuit << pq.RY(qubits[i], params[1][i])
        circuit << pq.RZ(qubits[i], params[2][i])
    return circuit

def vqc_circuit(inputs, param: Tensor, qubits, cubits, machine):
    circuit = pq.QCircuit()
    circuit << vqc_encoding(qubits, inputs)
    circuit << vqc_ansatz(qubits, param.reshape([3, -1]))
    vqc_prog = pq.QProg()
    vqc_prog << circuit
    pauli_dict = [{'Z0': 1}, {'Z1': 1}, {'Z2': 1}, {'Z3': 1}]
    return np.array([expval(machine, vqc_prog, p, qubits) for p in pauli_dict])

def vqc_lstm(inputs, weights, mapping={0: 45, 1: 46, 2: 52, 3: 53, 4: 54, 5: 48}):
    c = qpandalite.Circuit()
    # vqc encoding
    ry_params = inputs
    c.h(0)
    c.h(1)
    c.h(2)
    c.h(3)
    c.ry(0,ry_params[0])
    c.ry(1,ry_params[1])
    c.ry(2,ry_params[2])
    c.ry(3,ry_params[3])
    c.ry(0,ry_params[4])
    c.ry(1,ry_params[5])
    c.ry(2,ry_params[6])
    c.ry(3,ry_params[7])

    # for i,ry_param in enumerate(ry_params):
    #     c.ry(i, ry_param)
    #     # c.rx(i, ry_param)
    # # vqc_ansatz
    c.cx(0, 1)
    c.cx(1, 2)
    c.cx(2, 3)
    c.cx(3, 1)
    c.cx(0, 2)
    c.cx(1, 3)
    c.cx(2, 0)
    c.cx(3, 1)

    c.rz(0,weights[0])
    c.ry(0,weights[1])
    c.rz(0,weights[2])

    c.rz(0,weights[0])
    c.ry(0,weights[1])
    c.rz(0,weights[2])    
    
    c.rz(0,weights[0])
    c.ry(0,weights[1])
    c.rz(0,weights[2])    
    
    c.rz(0,weights[0])
    c.ry(0,weights[1])
    c.rz(0,weights[2])

    c.rz(0,weights[3])
    c.ry(0,weights[4])
    c.rz(0,weights[5])

    c.rz(0,weights[6])
    c.ry(0,weights[7])
    c.rz(0,weights[8])

    c.rz(0,weights[9])
    c.ry(0,weights[10])
    c.rz(0,weights[11])
    c.measure(0, 1, 2, 3)
    c = c.remapping(mapping)
    return c.circuit

class QLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz,vqc_circuit_lite, shots=2000):
        super().__init__()
    
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.n_qubits = hidden_sz
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.W = nn.Parameter((input_sz, hidden_sz * 4), dtype=dtype)
        self.U = nn.Parameter((hidden_sz, hidden_sz * 4), dtype=dtype)
        self.bias = nn.Parameter([hidden_sz * 4], dtype=dtype)
        
        self.vqc_forget = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['ZIIII', 'IZIII', 'IIZII','IIIZI','IIIIZ'],
                                                    is_dummy=DUMMY,
                                                    mapping=khan_mapping,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 3600, 'retry': 500}
                                                    )
        self.vqc_input = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['ZIIII', 'IZIII', 'IIZII','IIIZI','IIIIZ'],
                                                    is_dummy=DUMMY,
                                                    mapping=khan_mapping,  
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 3600, 'retry': 500}
                                                    )
        self.vqc_update = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['ZIIII', 'IZIII', 'IIZII','IIIZI','IIIIZ'],
                                                    is_dummy=DUMMY,
                                                    mapping=khan_mapping,  
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 3600, 'retry': 500}
                                                    )
        self.vqc_output = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['ZIIII', 'IZIII', 'IIZII','IIIZI','IIIIZ'],
                                                    is_dummy=DUMMY,
                                                    mapping=khan_mapping,  
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 3600, 'retry': 500}
                                                    )

        # self.vqc_forget = nn.Linear(hidden_sz,hidden_sz)
        # self.vqc_output = nn.Linear(hidden_sz,hidden_sz)
        # self.vqc_update = nn.Linear(hidden_sz,hidden_sz)
        # self.vqc_input = nn.Linear(hidden_sz,hidden_sz)
        # self.vqc_forget = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # self.vqc_input = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # self.vqc_update = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # self.vqc_output = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # 从头训练模型进行合理的参数初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.fill_rand_signed_uniform_(stdv)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        batch_sz, seq_sz, _ = x.shape
        hidden_seq = []
        if init_states is None:
            h_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True, dtype=dtype)
            c_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True, dtype=dtype)
        else:
            h_t, c_t = init_states
        hs = self.hidden_size
        for t in tqdm(range(seq_sz)):            
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            v_t = tensor.matmul(x_t, self.W) + tensor.matmul(h_t, self.U) + self.bias
            # encoding!!!!!!!!!!
            encoding_input = self.sigmoid(v_t) * np.pi
            # print(v_t)

            i_t = self.sigmoid(self.vqc_input(encoding_input[:, :hs]))
            f_t = self.sigmoid(self.vqc_forget(encoding_input[:, hs:2*hs]))
            g_t = self.tanh(self.vqc_update(encoding_input[:, 2*hs:3*hs]))
            o_t = self.sigmoid(self.vqc_output(encoding_input[:, 3*hs:]))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)

            hidden_seq.append(tensor.unsqueeze(h_t, 0))
        if seq_sz > 1:
            hidden_seq = tensor.concatenate(hidden_seq, 0)
            hidden_seq = hidden_seq.transpose([1, 0, 2])
        else:
            hidden_seq = tensor.unsqueeze(h_t, 0)
            
        return hidden_seq, (h_t, c_t)


def vqc_circuit_lite(inputs, weights, mapping={0: 45, 1: 46, 2: 52, 3: 53, 4: 54, 5: 48}):
    c = qpandalite.Circuit()
    # vqc encoding
    ry_params = inputs

    for i,ry_param in enumerate(ry_params):
        c.ry(i, ry_param)
        # c.rx(i, ry_param)
    # vqc_ansatz

    c.cz(0, 1)

    c.rx(0,weights[0])
    c.rx(1,weights[1])
    c.rx(2,weights[2])

    c.cz(1, 2)

    c.rx(0,weights[3])
    c.rx(1,weights[4])
    c.rx(2,weights[5])


    c.measure(0, 1, 2)
    c = c.remapping(mapping)
    return c.circuit


def vqc_circuit_lite_hard(inputs, weights, mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}):
    c = qpandalite.Circuit()
    # vqc encoding
    ry_params = inputs

    for i,ry_param in enumerate(ry_params):
        c.ry(i, ry_param)
        # c.rx(i, ry_param)
    # vqc_ansatz

    c.cz(0, 1)
    c.cz(2, 3)
    c.rx(0,weights[0])
    c.rx(1,weights[1])
    c.rx(2,weights[2])
    c.rx(3,weights[3])
    c.rx(4,weights[4])
    c.cz(1, 2)
    c.cz(3, 4)
    c.rx(2,weights[5])

    
    c.cz(0, 1)
    c.cz(2, 3)
    c.rx(0,weights[6])
    c.rx(1,weights[7])
    c.rx(2,weights[8])
    c.rx(3,weights[9])
    c.rx(4,weights[10])
    c.cz(1, 2)
    c.cz(3, 4)
    c.rx(0,weights[11])

    # c.cz(0, 1)
    # c.cz(2, 3)
    # c.rx(1,weights[16])
    # c.rx(2,weights[17])
    # c.rx(3,weights[18])
    # c.rx(4,weights[19])
    # c.cz(0, 1)
    # c.cz(2, 3)
    # c.rx(0,weights[20])
    # c.rx(1,weights[21])
    # c.rx(2,weights[22])
    # c.rx(3,weights[23])
    # c.rx(4,weights[24])
    # c.cz(1, 2)
    # c.cz(3, 4)
    # c.rx(0,weights[25])
    # c.rx(1,weights[26])
    # c.rx(2,weights[27])
    # c.rx(3,weights[28])
    # c.rx(4,weights[29])
    c.measure(0, 1, 2, 3 ,4 )
    c = c.remapping(mapping)
    return c.circuit

class RegLSTM(nn.Module):
    """
    constructor
    :param input_sz: num of features
    :param hidden_sz: num of hidden neurons
    """

    def __init__(self, input_sz, hidden_sz,vqc_circuit_lite = vqc_circuit_lite):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.ReLu = nn.ReLu()
        self.Tanh = nn.Tanh()
        self.rnn = QLSTM(input_sz, hidden_sz, vqc_circuit_lite)
        self.cls = nn.Linear(hidden_sz, 2, dtype=dtype)

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        output, (h_n, _) = self.rnn(x)
        output = self.cls(h_n)
        return output
    
class RegLSTM_hard(nn.Module):
    """
    constructor
    :param input_sz: num of features
    :param hidden_sz: num of hidden neurons
    """

    def __init__(self, input_sz, hidden_sz,vqc_circuit_lite = vqc_circuit_lite_hard):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.ReLu = nn.ReLu()
        self.Tanh = nn.Tanh()
        self.rnn = QLSTM(input_sz, hidden_sz, vqc_circuit_lite)
        self.cls = nn.Linear(hidden_sz, 3, dtype=dtype)

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        output, (h_n, _) = self.rnn(x)
        output = self.cls(h_n)
        return output

class ClassicLSTM(nn.Module):
    """
    constructor
    :param input_sz: num of features
    :param hidden_sz: num of hidden neurons
    """

    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.rnn = Classic_LSTM_naf(input_sz, hidden_sz)
        self.cls = nn.Linear(4, 2, dtype=dtype)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.fill_rand_signed_uniform_(stdv)

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        output, (h_n, _) = self.rnn(x)
        output = self.cls(h_n)
        # output = nn.Softmax()(output)
        return output

class QLSTM_8vqc(nn.Module):
    def __init__(self, input_sz, hidden_sz, vqc_circuit_lite , shots=2000):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.n_qubits = hidden_sz
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.W = nn.Parameter((input_sz, hidden_sz * 8), dtype=dtype)
        self.U = nn.Parameter((hidden_sz, hidden_sz * 8), dtype=dtype)
        self.bias = nn.Parameter([hidden_sz * 8], dtype=dtype)

        self.vqc_forget_h = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )    
        self.vqc_forget_x = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )        
        self.vqc_input_h = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )       
        self.vqc_input_x = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )
        self.vqc_update_h = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )
        self.vqc_update_x = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )
        self.vqc_output_h = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )
        
        self.vqc_output_x = qnn.QuantumLayerWITHQLITE(vqc_circuit_lite,
                                                    para_num=self.n_qubits * 3,
                                                    hamiltonian=['IIIZ', 'IIZI', 'IZII', 'ZIII'],
                                                    is_dummy=DUMMY,
                                                    submit_kwargs={'shots':shots, 'auto_mapping': False},
                                                    query_kwargs={'timeout': 24 * 3600, 'interval': 60, 'retry': 1000}
                                                    )

        # self.vqc_forget = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # self.vqc_input = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # self.vqc_update = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # self.vqc_output = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, "cpu", self.n_qubits, self.n_qubits)
        # 从头训练模型进行合理的参数初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.fill_rand_signed_uniform_(stdv)

    def forward(self, x, init_states=None):

        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.shape
        hidden_seq = []
        if init_states is None:
            h_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True, dtype=dtype)
            c_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True, dtype=dtype)
        else:
            h_t, c_t = init_states
        hs = self.hidden_size

        for t in range(seq_sz):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            v_t = tensor.matmul(x_t, self.W) + tensor.matmul(h_t, self.U) + self.bias

            i_t_h = self.sigmoid(self.vqc_input_h(v_t[:, :hs]))
            i_t_x = self.sigmoid(self.vqc_input_x(v_t[:, hs:2*hs]))
            i_t = i_t_h + i_t_x

            f_t_h = self.sigmoid(self.vqc_forget_h(v_t[:,2*hs:3*hs]))
            f_t_x = self.sigmoid(self.vqc_forget_x(v_t[:,3*hs:4*hs]))
            f_t = f_t_h + f_t_x

            g_t_h = self.tanh(self.vqc_update_h(v_t[:, 4*hs:5*hs]))
            g_t_x = self.tanh(self.vqc_update_x(v_t[:, 5*hs:6*hs]))
            g_t = g_t_h + g_t_x

            o_t_h = self.sigmoid(self.vqc_output_h(v_t[:, 6*hs:7*hs]))
            o_t_x = self.sigmoid(self.vqc_output_x(v_t[:, 7*hs:8*hs]))
            o_t = o_t_h + o_t_x
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            hidden_seq.append(tensor.unsqueeze(h_t, 0))
        if seq_sz > 1:
            hidden_seq = tensor.concatenate(hidden_seq, 0)
            hidden_seq = hidden_seq.transpose([1, 0, 2])
        else:
            hidden_seq = tensor.unsqueeze(h_t, 0)
        return hidden_seq, (h_t, c_t)

class Classic_LSTM_naf(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter((input_sz, hidden_sz * 4), dtype=dtype)
        self.U = nn.Parameter((hidden_sz, hidden_sz * 4), dtype=dtype)
        self.bias = nn.Parameter([hidden_sz * 4], dtype=dtype)
        self.f_gate = nn.Linear(hidden_sz,4,dtype = dtype)
        self.i_gate = nn.Linear(hidden_sz,4,dtype = dtype)
        self.g_gate = nn.Linear(hidden_sz,4,dtype = dtype)
        self.o_gate = nn.Linear(hidden_sz,4,dtype = dtype)
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh() 
    def forward(self, x, init_states=None):
        batch_sz, seq_sz, _ = x.shape
        hidden_seq = []
        if init_states is None:
            h_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True, dtype=dtype)
            c_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True, dtype=dtype)
        else:
            h_t, c_t = init_states
        hs = self.hidden_size

        for t in range(seq_sz):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            v_t = tensor.matmul(x_t, self.W) + tensor.matmul(h_t, self.U) + self.bias

            
            i_t = self.Sigmoid(self.i_gate(v_t[:, :hs]))
            f_t = self.Sigmoid(self.f_gate(v_t[:, hs:2*hs]))
            g_t = self.Tanh(self.g_gate(v_t[:,2*hs:3*hs]))
            o_t = self.Sigmoid(self.o_gate(v_t[:,3*hs:4*hs]))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.Tanh(c_t)
            hidden_seq.append(tensor.unsqueeze(h_t, 0))
        if seq_sz > 1:
            hidden_seq = tensor.concatenate(hidden_seq, 0)
            hidden_seq = hidden_seq.transpose([1, 0, 2])
        else:
            hidden_seq = tensor.unsqueeze(h_t, 0)
        return hidden_seq, (h_t, c_t)
