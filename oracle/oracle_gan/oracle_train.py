import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from utils.metrics.Nll import Nll
from utils.metrics.DocEmbSim import DocEmbSim
from utils.utils import *
from utils.ops import gradient_penalty

EPS = 1e-10


def oracle_train(generator, discriminator, oracle_model, oracle_loader, gen_loader, config):
    batch_size = config['batch_size']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    num_sentences = config['num_sentences']
    data_dir = config['data_dir']
    log_dir = config['log_dir']
    sample_dir = config['sample_dir']
    npre_epochs = config['npre_epochs']
    nadv_steps = config['nadv_steps']
    seed = config['seed']
    temper = config['temperature']
    adapt = config['adapt']

    # set random seed
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # filename
    oracle_file = os.path.join(sample_dir, 'oracle.txt')
    gen_file = os.path.join(sample_dir, 'generator.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-relgan.csv')

    # create necessary directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # placeholder definitions
    x_real = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_real")  # tokens of oracle sequences

    temperature = tf.Variable(1., trainable=False, name='temperature')

    x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    x_fake_onehot_appr, x_fake, g_pretrain_loss, gen_o = generator(x_real=x_real, temperature=temperature)

    d_out_real = discriminator(x_onehot=x_real_onehot)
    d_out_fake = discriminator(x_onehot=x_fake_onehot_appr)

    # GAN / Divergence type
    log_pg, g_loss, d_loss = get_losses(d_out_real, d_out_fake, x_real_onehot, x_fake_onehot_appr,
                                                    gen_o, discriminator, config)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Train ops
    g_pretrain_op, g_train_op, d_train_op, temp_train_op = get_train_ops(config, g_pretrain_loss, g_loss, d_loss,
                                                                         log_pg, temperature, global_step)

    # Record wall clock time
    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    temp_var = tf.placeholder(tf.float32)
    update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
        tf.summary.scalar('loss/discriminator', d_loss),
        tf.summary.scalar('loss/g_loss', g_loss),
        tf.summary.scalar('loss/log_pg', log_pg),
        tf.summary.scalar('loss/Wall_clock_time', Wall_clock_time),
        tf.summary.scalar('loss/temperature', temperature),
    ]
    loss_summary_op = tf.summary.merge(loss_summaries)

    # Metric Summaries
    metrics_pl, metric_summary_op = get_metric_summary_op(config)

    # ------------- initial the graph --------------
    with init_sess() as sess:
        log = open(csv_file, 'w')
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'), sess.graph)

        # generate oracle data and create batches
        generate_samples(sess, oracle_model.gen_x, batch_size, num_sentences, oracle_file)
        oracle_loader.create_batches(oracle_file)

        metrics = get_metrics(config, oracle_loader, gen_loader, oracle_file, gen_file,
                              oracle_model, g_pretrain_loss, x_real, sess)

        print('Start pre-training...')
        for epoch in range(npre_epochs):
            # run pre-training
            g_pretrain_loss_np = pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, oracle_loader)

            # Test
            ntest_pre = 10
            if np.mod(epoch, ntest_pre) == 0:
                # generate fake data and create batches
                gen_save_file = os.path.join(sample_dir, 'pre_samples_{:05d}.txt'.format(epoch))
                generate_samples(sess, x_fake, batch_size, num_sentences, gen_file)
                generate_samples(sess, x_fake, batch_size, 120, gen_save_file)
                gen_loader.create_batches(gen_file)

                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, epoch)

                msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    msg += ', ' + name + ': %.4f' % score
                print(msg)
                log.write(msg)
                log.write('\n')

        print('Start adversarial training...')
        progress = tqdm(range(nadv_steps))
        for _ in progress:
            niter = sess.run(global_step)

            t0 = time.time()

            # run adversarial training
            for _ in range(config['gsteps']):
                sess.run(g_train_op, feed_dict={x_real: oracle_loader.random_batch()})
            for _ in range(config['dsteps']):
                sess.run(d_train_op, feed_dict={x_real: oracle_loader.random_batch()})

            t1 = time.time()
            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            # temperature
            temp_var_np = get_fixed_temperature(temper, niter, nadv_steps, adapt)
            sess.run(update_temperature_op, feed_dict={temp_var: temp_var_np})

            feed = {x_real: oracle_loader.random_batch()}
            g_loss_np, d_loss_np, loss_summary_str = sess.run([g_loss, d_loss, loss_summary_op], feed_dict=feed)
            sum_writer.add_summary(loss_summary_str, niter)

            sess.run(global_step_op)

            progress.set_description('g_loss: %4.4f, d_loss: %4.4f' % (g_loss_np, d_loss_np))

            # Test
            if np.mod(niter, config['ntest']) == 0:
                # generate fake data and create batches
                gen_save_file = os.path.join(sample_dir, 'adv_samples_{:05d}.txt'.format(niter))
                generate_samples(sess, x_fake, batch_size, num_sentences, gen_file)
                generate_samples(sess, x_fake, batch_size, 120, gen_save_file)
                gen_loader.create_batches(gen_file)

                # write summaries
                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, niter + config['npre_epochs'])

                msg = 'adv_step: ' + str(niter)
                metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    msg += ', ' + name + ': %.4f' % score
                print(msg)
                log.write(msg)
                log.write('\n')


# A function to get different GAN losses
def get_losses(d_out_real, d_out_fake, x_real_onehot, x_fake_onehot_appr, gen_o, discriminator, config):
    batch_size = config['batch_size']
    gan_type = config['gan_type']

    if gan_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.ones_like(d_out_fake)
        ))

    elif gan_type == 'JS':  # the vanilla GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(-d_out_fake)

    elif gan_type == 'hinge':  # the hinge loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'tv':  # the total variation distance
        d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
        g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))

    elif gan_type == 'wgan-gp': # WGAN-GP
        d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
        GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
        d_loss += GP

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'LS': # LS-GAN
        d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, 1.0))
        d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, 1.0))

    elif gan_type == 'RSGAN':  # relativistic standard GAN
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
        ))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
        ))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)

    return log_pg, g_loss, d_loss


# A function to calculate the gradients and get training operations
def get_train_ops(config, g_pretrain_loss, g_loss, d_loss, log_pg, temperature, global_step):
    optimizer_name = config['optimizer']
    nadv_steps = config['nadv_steps']
    d_lr = config['d_lr']
    gpre_lr = config['gpre_lr']
    gadv_lr = config['gadv_lr']

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    grad_clip = 5.0  # keep the same with the previous setting

    # generator pre-training
    pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999)
    pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars), grad_clip)  # gradient clipping
    g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

    if config['decay']:
        d_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        gadv_lr = tf.train.exponential_decay(gadv_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)

    if optimizer_name == 'adam':
        d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        g_optimizer = tf.train.AdamOptimizer(gadv_lr, beta1=0.9, beta2=0.999)
        temp_optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.9, beta2=0.999)
    elif optimizer_name == 'rmsprop':
        d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        g_optimizer = tf.train.RMSPropOptimizer(gadv_lr)
        temp_optimizer = tf.train.RMSPropOptimizer(1e-2)
    else:
        raise NotImplementedError

    g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars), grad_clip)  # gradient clipping
    g_train_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))

    print('len of g_grads without None: {}'.format(len([i for i in g_grads if i is not None])))
    print('len of g_grads: {}'.format(len(g_grads)))

    # d_train_op = d_optimizer.minimize(d_loss, var_list=d_vars)
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss, d_vars), grad_clip)  # gradient clipping
    d_train_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))

    temp_grads = tf.gradients(-log_pg, [temperature])
    temp_train_op = temp_optimizer.apply_gradients(zip(temp_grads, [temperature]))

    return g_pretrain_op, g_train_op, d_train_op, temp_train_op


# A function to get various evaluation metrics
def get_metrics(config, oracle_loader, gen_loader, oracle_file, gen_file, oracle_model, g_pretrain_loss, x_real, sess):
    # set up evaluation metric
    metrics = []
    if config['nll_oracle']:
        nll_oracle = Nll(gen_loader, oracle_model.pretrain_loss, oracle_model.x, sess, name='nll_oracle')
        metrics.append(nll_oracle)
    if config['nll_gen']:
        nll_gen = Nll(oracle_loader, g_pretrain_loss, x_real, sess, name='nll_gen')
        metrics.append(nll_gen)
    if config['doc_embsim']:
        doc_embsim = DocEmbSim(oracle_file, gen_file, config['vocab_size'], name='doc_embsim')
        metrics.append(doc_embsim)

    return metrics


# A function to get the summary for each metric
def get_metric_summary_op(config):
    metrics_pl = []
    metrics_sum = []
    if config['nll_oracle']:
        nll_oracle = tf.placeholder(tf.float32)
        metrics_pl.append(nll_oracle)
        metrics_sum.append(tf.summary.scalar('metrics/nll_oracle', nll_oracle))

    if config['nll_gen']:
        nll_gen = tf.placeholder(tf.float32)
        metrics_pl.append(nll_gen)
        metrics_sum.append(tf.summary.scalar('metrics/nll_gen', nll_gen))

    if config['doc_embsim']:
        doc_embsim = tf.placeholder(tf.float32)
        metrics_pl.append(doc_embsim)
        metrics_sum.append(tf.summary.scalar('metrics/doc_embsim', doc_embsim))

    metric_summary_op = tf.summary.merge(metrics_sum)
    return metrics_pl, metric_summary_op


def get_fixed_temperature(temper, i, nadv_steps, adapt):
    # using a fixed number of maximum adversarial steps
    N = 5000
    assert nadv_steps <= N
    if adapt == 'no':
        temper_var_np = temper  # no increase
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1)**2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np
