import math
import tensorflow as tf


def hw_flatten(x):
    return tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2], x.shape[-1]])


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def create_linear_initializer(input_size, dtype=tf.float32):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1 / math.sqrt(input_size * 1.0)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def create_bias_initializer(dtype=tf.float32):
    """Returns a default initializer for the biases of a linear/AddBias module."""
    return tf.zeros_initializer(dtype=dtype)


def linear(input_, output_size, use_bias=False, sn=False, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: Variable Scope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        W = tf.get_variable("Matrix", shape=[output_size, input_size],
                            initializer=create_linear_initializer(input_size, input_.dtype),
                            dtype=input_.dtype)
        if sn:
            W = spectral_norm(W)
        output_ = tf.matmul(input_, tf.transpose(W))
        if use_bias:
            bias_term = tf.get_variable("Bias", [output_size],
                                        initializer=create_bias_initializer(input_.dtype),
                                        dtype=input_.dtype)
            output_ += bias_term

    return output_


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def mlp(input_, output_sizes, act_func=tf.nn.relu, use_bias=True):
    '''
    Constructs a MLP module
    :param input_:
    :param output_sizes: An iterable of output dimensionalities
    :param act_func: activation function
    :param use_bias: whether use bias term for linear mapping
    :return: the output of the MLP module
    '''
    net = input_
    num_layers = len(output_sizes)
    for layer_id in range(num_layers):
        net = linear(net, output_sizes[layer_id], use_bias=use_bias, scope='linear_{}'.format(layer_id))
        if layer_id != num_layers - 1:
            net = act_func(net)
    return net


def conv2d(input_, out_nums, k_h=2, k_w=1, d_h=2, d_w=1, stddev=None, sn=False, padding='SAME', scope=None):
    in_nums = input_.get_shape().as_list()[-1]
    # Glorot initialization
    if stddev is None:
        stddev = math.sqrt(2. / (k_h * k_w * in_nums))
    with tf.variable_scope(scope or "Conv2d"):
        W = tf.get_variable("Matrix", shape=[k_h, k_w, in_nums, out_nums],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            W = spectral_norm(W)
        b = tf.get_variable("Bias", shape=[out_nums], initializer=tf.zeros_initializer)
        conv = tf.nn.conv2d(input_, filter=W, strides=[1, d_h, d_w, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)

    return conv


def self_attention(x, ch, sn=False):
    """self-attention for GAN"""
    f = conv2d(x, ch // 8, k_h=1, d_h=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
    g = conv2d(x, ch // 8, k_h=1, d_h=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
    h = conv2d(x, ch, k_h=1, d_h=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

    beta = tf.nn.softmax(s, dim=-1)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, [-1] + x.get_shape().as_list()[1:])  # [bs, h, w, C]
    x = gamma * o + x

    return x


def spectral_norm(w, iteration=1):
    """spectral normalization for GANs"""
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def create_output_unit(output_size, vocab_size):
    # output_size = self.gen_mem.output_size.as_list()[0]
    Wo = tf.get_variable('Wo', shape=[output_size, vocab_size], initializer=create_linear_initializer(output_size))
    bo = tf.get_variable('bo', shape=[vocab_size], initializer=create_bias_initializer())

    def unit(hidden_mem_o):
        logits = tf.matmul(hidden_mem_o, Wo) + bo
        return logits

    return unit


def add_gumbel(o_t, eps=1e-10):
    """Sample from Gumbel(0, 1)"""
    u = tf.random_uniform(tf.shape(o_t), minval=0, maxval=1, dtype=tf.float32)
    g_t = -tf.log(-tf.log(u + eps) + eps)
    gumbel_t = tf.add(o_t, g_t)
    return gumbel_t


def add_gumbel_cond(o_t, next_token_onehot, eps=1e-10):
    """draw reparameterization z of categorical variable b from p(z|b)."""

    def truncated_gumbel(gumbel, truncation):
        return -tf.log(eps + tf.exp(-gumbel) + tf.exp(-truncation))

    v = tf.random_uniform(tf.shape(o_t), minval=0, maxval=1, dtype=tf.float32)

    print("shape of v: {}".format(v.get_shape().as_list()))
    print("shape of next_token_onehot: {}".format(next_token_onehot.get_shape().as_list()))

    gumbel = -tf.log(-tf.log(v + eps) + eps, name="gumbel")
    topgumbels = gumbel + tf.reduce_logsumexp(o_t, axis=-1, keep_dims=True)
    topgumbel = tf.reduce_sum(next_token_onehot * topgumbels, axis=-1, keep_dims=True)

    truncgumbel = truncated_gumbel(gumbel + o_t, topgumbel)
    return (1. - next_token_onehot) * truncgumbel + next_token_onehot * topgumbels


def gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config):
    """compute the gradiet penalty for the WGAN-GP loss"""
    alpha = tf.random_uniform(shape=[config['batch_size'], 1, 1], minval=0., maxval=1.)
    interpolated = alpha * x_real_onehot + (1. - alpha) * x_fake_onehot_appr

    logit = discriminator(x_onehot=interpolated)

    grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
    grad_norm = tf.norm(tf.layers.flatten(grad), axis=1)  # l2 norm

    GP = config['reg_param'] * tf.reduce_mean(tf.square(grad_norm - 1.))

    return GP
