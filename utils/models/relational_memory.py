"""Relational Memory architecture.

An implementation of the architecture described in "Relational Recurrent
Neural Networks", Santoro et al., 2018.
"""
import tensorflow as tf
from utils.ops import linear, mlp


class RelationalMemory(object):
    """Relational Memory Core."""

    def __init__(self, mem_slots, head_size, num_heads=1, num_blocks=1,
                 forget_bias=1.0, input_bias=0.0, gate_style='unit',
                 attention_mlp_layers=2, key_size=None, name='relational_memory'):
        """Constructs a `RelationalMemory` object.

        Args:
          mem_slots: The total number of memory slots to use.
          head_size: The size of an attention head.
          num_heads: The number of attention heads to use. Defaults to 1.
          num_blocks: Number of times to compute attention per time step. Defaults
            to 1.
          forget_bias: Bias to use for the forget gate, assuming we are using
            some form of gating. Defaults to 1.
          input_bias: Bias to use for the input gate, assuming we are using
            some form of gating. Defaults to 0.
          gate_style: Whether to use per-element gating ('unit'),
            per-memory slot gating ('memory'), or no gating at all (None).
            Defaults to `unit`.
          attention_mlp_layers: Number of layers to use in the post-attention
            MLP. Defaults to 2.
          key_size: Size of vector to use for key & query vectors in the attention
            computation. Defaults to None, in which case we use `head_size`.
          name: Name of the module.

        Raises:
          ValueError: gate_style not one of [None, 'memory', 'unit'].
          ValueError: num_blocks is < 1.
          ValueError: attention_mlp_layers is < 1.
        """

        self._mem_slots = mem_slots
        self._head_size = head_size
        self._num_heads = num_heads
        self._mem_size = self._head_size * self._num_heads
        self._name = name

        if num_blocks < 1:
            raise ValueError('num_blocks must be >= 1. Got: {}.'.format(num_blocks))
        self._num_blocks = num_blocks

        self._forget_bias = forget_bias
        self._input_bias = input_bias

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. Got: '
                '{}.'.format(gate_style))
        self._gate_style = gate_style

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self._attention_mlp_layers = attention_mlp_layers

        self._key_size = key_size if key_size else self._head_size

        self._template = tf.make_template(self._name, self._build)  # wrapper for variable sharing

    def initial_state(self, batch_size):
        """Creates the initial memory.

        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self._mem_slots, self._mem_size).

        Args:
          batch_size: The size of the batch.

        Returns:
          init_state: A truncated or padded matrix of size
            (batch_size, self._mem_slots, self._mem_size).
        """
        init_state = tf.eye(self._mem_slots, batch_shape=[batch_size])

        # Pad the matrix with zeros.
        if self._mem_size > self._mem_slots:
            difference = self._mem_size - self._mem_slots
            pad = tf.zeros((batch_size, self._mem_slots, difference))
            init_state = tf.concat([init_state, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif self._mem_size < self._mem_slots:
            init_state = init_state[:, :, :self._mem_size]
        return init_state

    def _multihead_attention(self, memory):
        """Perform multi-head attention from 'Attention is All You Need'.

        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.

        Args:
          memory: Memory tensor to perform attention on, with size [B, N, H*V].

        Returns:
          new_memory: New memory tensor.
        """

        qkv_size = 2 * self._key_size + self._head_size
        total_size = qkv_size * self._num_heads  # Denote as F.
        batch_size = memory.get_shape().as_list()[0]  # Denote as B
        memory_flattened = tf.reshape(memory, [-1, self._mem_size])  # [B * N, H * V]
        qkv = linear(memory_flattened, total_size, use_bias=False, scope='lin_qkv')  # [B*N, F]
        qkv = tf.reshape(qkv, [batch_size, -1, total_size])  # [B, N, F]
        qkv = tf.contrib.layers.layer_norm(qkv, trainable=True)  # [B, N, F]

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = tf.reshape(qkv, [batch_size, -1, self._num_heads, qkv_size])

        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(qkv_transpose, [self._key_size, self._key_size, self._head_size], -1)

        q *= qkv_size ** -0.5
        dot_product = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]
        weights = tf.nn.softmax(dot_product)

        output = tf.matmul(weights, v)  # [B, H, N, V]

        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])

        # [B, N, H, V] -> [B, N, H * V]
        new_memory = tf.reshape(output_transpose, [batch_size, -1, self._mem_size])
        return new_memory

    @property
    def state_size(self):
        return tf.TensorShape([self._mem_slots, self._mem_size])

    @property
    def output_size(self):
        return tf.TensorShape(self._mem_slots * self._mem_size)

    def _calculate_gate_size(self):
        """Calculate the gate size from the gate_style.

        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self._gate_style == 'unit':
            return self._mem_size
        elif self._gate_style == 'memory':
            return 1
        else:  # self._gate_style == None
            return 0

    def _create_gates(self, inputs, memory):
        """Create input and forget gates for this step using `inputs` and `memory`.

        Args:
          inputs: Tensor input.
          memory: The current state of memory.

        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.
        num_gates = 2 * self._calculate_gate_size()
        batch_size = memory.get_shape().as_list()[0]

        memory = tf.tanh(memory)  # B x N x H * V

        inputs = tf.reshape(inputs, [batch_size, -1])  # B x In_size
        gate_inputs = linear(inputs, num_gates, use_bias=False, scope='gate_in')  # B x num_gates
        gate_inputs = tf.expand_dims(gate_inputs, axis=1)  # B x 1 x num_gates

        memory_flattened = tf.reshape(memory, [-1, self._mem_size])  # [B * N, H * V]
        gate_memory = linear(memory_flattened, num_gates, use_bias=False, scope='gate_mem')  # [B * N, num_gates]
        gate_memory = tf.reshape(gate_memory, [batch_size, self._mem_slots, num_gates])  # [B, N, num_gates]

        gates = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=2)
        input_gate, forget_gate = gates  # B x N x num_gates/2, B x N x num_gates/2

        input_gate = tf.sigmoid(input_gate + self._input_bias)
        forget_gate = tf.sigmoid(forget_gate + self._forget_bias)

        return input_gate, forget_gate

    def _attend_over_memory(self, memory):
        """Perform multiheaded attention over `memory`.

        Args:
          memory: Current relational memory.

        Returns:
          The attended-over memory.
        """

        for _ in range(self._num_blocks):
            attended_memory = self._multihead_attention(memory)  # [B, N, H * V]

            # Add a skip connection to the multiheaded attention's input.
            memory = tf.contrib.layers.layer_norm(memory + attended_memory, trainable=True)  # [B, N, H * V]

            # Add a mlp map
            batch_size = memory.get_shape().as_list()[0]

            memory_mlp = tf.reshape(memory, [-1, self._mem_size])  # [B * N, H * V]
            memory_mlp = mlp(memory_mlp, [self._mem_size] * self._attention_mlp_layers)  # [B * N, H * V]
            memory_mlp = tf.reshape(memory_mlp, [batch_size, -1, self._mem_size])

            # Add a skip connection to the memory_mlp's input.
            memory = tf.contrib.layers.layer_norm(memory + memory_mlp, trainable=True)  # [B, N, H * V]

        return memory

    def _build(self, inputs, memory):
        """Adds relational memory to the TensorFlow graph.

        Args:
          inputs: Tensor input.
          memory: Memory output from the previous time step.

        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """

        batch_size = memory.get_shape().as_list()[0]
        inputs = tf.reshape(inputs, [batch_size, -1])  # [B, In_size]
        inputs = linear(inputs, self._mem_size, use_bias=True, scope='input_for_cancat')  # [B, V * H]
        inputs_reshape = tf.expand_dims(inputs, 1)  # [B, 1, V * H]

        memory_plus_input = tf.concat([memory, inputs_reshape], axis=1)  # [B, N + 1, V * H]
        next_memory = self._attend_over_memory(memory_plus_input)  # [B, N + 1, V * H]

        n = inputs_reshape.get_shape().as_list()[1]
        next_memory = next_memory[:, :-n, :]  # [B, N, V * H]

        if self._gate_style == 'unit' or self._gate_style == 'memory':
            self._input_gate, self._forget_gate = self._create_gates(inputs_reshape, memory)
            next_memory = self._input_gate * tf.tanh(next_memory)
            next_memory += self._forget_gate * memory

        output = tf.reshape(next_memory, [batch_size, -1])
        return output, next_memory

    def __call__(self, *args, **kwargs):
        """Operator overload for calling.

        This is the entry point when users connect a Module into the Graph. The
        underlying _build method will have been wrapped in a Template by the
        constructor, and we call this template with the provided inputs here.

        Args:
          *args: Arguments for underlying _build method.
          **kwargs: Keyword arguments for underlying _build method.

        Returns:
          The result of the underlying _build method.
        """
        outputs = self._template(*args, **kwargs)

        return outputs

    @property
    def input_gate(self):
        """Returns the input gate Tensor."""
        return self._input_gate

    @property
    def forget_gate(self):
        """Returns the forget gate Tensor."""
        return self._forget_gate

    @property
    def rmc_params(self):
        """Returns the parameters in the RMC module"""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

    def set_rmc_params(self, ref_rmc_params):
        """Set parameters of the RMC module to be the same with those of the reference module"""
        rmc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        if len(rmc_params) != len(ref_rmc_params):
            raise ValueError("the number of parameters in the two RMC modules does not match")
        for i in range(len(ref_rmc_params)):
            rmc_params[i] = tf.identity(ref_rmc_params[i])

    def update_rmc_params(self, ref_rmc_params, update_ratio):
        """Update parameters of the RMC module based on a reference module"""
        rmc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        if len(rmc_params) != len(ref_rmc_params):
            raise ValueError("the number of parameters in the two RMC modules does not match")
        for i in range(len(ref_rmc_params)):
            rmc_params[i] = update_ratio * rmc_params[i] + (1 - update_ratio) * tf.identity(ref_rmc_params[i])
