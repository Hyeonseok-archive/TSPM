SEQ_LEN = 24 * 7
PRED_LEN = 24 * 7
SEG_LEN = 24
# # if floor == 1~2
# N_PLACE = 4
# if floor == 3~7
N_PLACE = 4

DROPOUT = 0.0

# NLinear DLinear params
class LinearConfig:
    def __init__(self):
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.c_in = N_PLACE
        self.mode='multi'

# SegRNN params
class SegRNNConfig:
    def __init__(self):
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.enc_in = N_PLACE
        ## hidden_dim
        self.d_model = 64
        self.seg_len = SEG_LEN
        self.dropout = DROPOUT
        self.rnn_type='gru'
        self.dec_way='pmf'
        self.channel_id=True
        self.revin=True


# TimesNet params
class TimesNetConfig:
    def __init__(self, seq_len=24, pred_len=24, enc_in=4, c_out=4):
    # def __init__(self, seq_len=24, label_len=24, pred_len=24, enc_in=4, c_out=4):
        self.task_name = 'forecasting'
        self.seq_len = seq_len
        self.label_len = pred_len
        self.pred_len = pred_len
        # num of timesblock
        self.e_layers = 3
        ## hidden, embedding_dim
        # timesblock
        self.d_ff = 128
        # data embedding
        self.enc_in = enc_in
        self.d_model = 64
        self.embed = 64
        self.freq = 64
        self.top_k = 2
        self.num_kernels = 3
        self.dropout = 0
        self.c_out = c_out

# PatchTST params
class PatchTSTConfig:
    def __init__(self, seq_len=24, seg_len=12, enc_in=4):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = seq_len
        self.e_layers = 3
        self.n_heads = 8
        self.d_model = 64
        self.d_ff = 64
        self.dropout = 0
        self.fc_dropout = 0
        self.head_dropout = 0
        self.individual = False
        self.patch_len = seg_len
        self.stride = 1
        self.padding_patch = None
        self.revin = True
        self.affine = True
        self.subtract_last = False
        self.decomposition = True
        self.kernel_size = 3


# SparseTSF params
class SparseTSFConfig:
    def __init__(self, seq_len=24, enc_in=4, period_len=12):
        self.seq_len = seq_len
        self.pred_len = seq_len
        self.enc_in = enc_in
        self.period_len = period_len


# ST params
class STConfig:
    def __init__(self):
        self.in_shape = (SEQ_LEN, 12, 20)
        self.s_hid = 64
        self.t_hid = 512
        self.n_section = N_PLACE
        self.drop_path = DROPOUT
        self.init_value = 1e-2