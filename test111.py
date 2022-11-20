class Conv1D(fluid.dygraph.Layer):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, act: str = None, dilation_rate: int = 1):
        super(Conv1D, self).__init__(None)
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.rec_field = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)
        self.pad = self.rec_field // 2
        self.conv1d = fluid.dygraph.Conv2D(num_channels=1, num_filters=output_dim, filter_size=(3, input_dim),
                                           padding=(self.pad, 0), dilation=(dilation_rate, 1), act=act)

    def forward(self, seq):
        h = fluid.layers.unsqueeze(seq, axes=[1])
        h = self.conv1d(h)
        h = fluid.layers.squeeze(h, axes=[3])
        h = fluid.layers.transpose(h, perm=[0, 2, 1])
        return h


class GatedDilatedResidualConv1D(fluid.dygraph.Layer):
    def __init__(self, dim: int, dilation_rate: int):
        super(GatedDilatedResidualConv1D, self).__init__(None)
        self.dim = dim
        self.conv1d = Conv1D(input_dim=self.dim, output_dim=2 * self.dim, kernel_size=3, dilation_rate=dilation_rate)

    def forward(self, seq, mask):
        c = self.conv1d(seq)

        def _gate(x):
            dropout_rate = 0.1
            s, h = x
            g, h = h[:, :, :self.dim], h[:, :, self.dim:]
            g = fluid.layers.dropout(g, dropout_rate, dropout_implementation="upscale_in_train")
            g = fluid.layers.sigmoid(g)
            return g * s + (1 - g) * h

        seq = _gate([seq, c])
        seq = seq * mask
        return seq