import tensorflow as tf
import os, glob
import argparse
from utils.utils import *
import models
from real.real_gan.real_train import get_metrics
from real.real_gan.real_loader import RealDataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Train and run a RmcGAN')

parser.add_argument('--hidden-dim', default=32, type=int, help='only used for OrcaleLstm and lstm_vanilla (generator)')
parser.add_argument('--ckpt-dir', default='', type=str, help='checkpoint dir')
parser.add_argument('--g-architecture', default='rmc_vanilla', type=str, help='Architecture for generator')
parser.add_argument('--vocab-size', default=5000, type=int, help="vocabulary size")
parser.add_argument('--seq-len', default=20, type=int, help="sequence length: [20, 40]")
parser.add_argument('--start-token', default=0, type=int, help="start token for a sentence")
parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
parser.add_argument('--mem-slots', default=1, type=int, help="memory size")
parser.add_argument('--head-size', default=512, type=int, help="head size or memory size")
parser.add_argument('--num-heads', default=2, type=int, help="number of heads")
parser.add_argument('--gen-emb-dim', default=32, type=int, help="generator embedding dimension")
parser.add_argument('--num-sentences', default=10000, type=int, help="number of total sentences")

# evaluation
parser.add_argument('--nll-oracle', default=False, action='store_true', help='if using nll-oracle metric')
parser.add_argument('--nll-gen', default=False, action='store_true', help='if using nll-gen metric')
parser.add_argument('--bleu', default=False, action='store_true', help='if using bleu metric, [2,3,4,5]')
parser.add_argument('--selfbleu', default=False, action='store_true', help='if using selfbleu metric, [2,3,4,5]')
parser.add_argument('--doc-embsim', default=False, action='store_true', help='if using DocEmbSim metric')

args = parser.parse_args()
pp.pprint(vars(args))
config = vars(args)

# paths and files
dataset = 'image_coco'
checkpoint_dir = 'real/experiments/out/20190715/image_coco/image_coco_rmc_vanilla_' \
                 'RSGAN_adam_bs64_sl20_sn0_dec0_ad-exp_npre150_nadv2000_ms1_hs256_nh2_' \
                 'ds5_dlr1e-4_glr1e-4_tem100_demb64_nrep64_hdim32_sd172/tf_logs/ckpt'
meta_file = glob.glob(os.path.join(checkpoint_dir, '*.meta'))[-1]

test_samples_dir = os.path.join(checkpoint_dir, 'samples')
test_gen_file = os.path.join(test_samples_dir, 'generator.txt')
data_file = os.path.join('data', '{}.txt'.format(dataset))
oracle_file = os.path.join(test_samples_dir, 'oracle_{}.txt'.format(dataset))
test_file = os.path.join('data', 'testdata/test_coco.txt')

if not os.path.exists(test_samples_dir):
    os.makedirs(test_samples_dir)

seq_len, vocab_size = text_precess(data_file)
print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))

generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                 seq_len=seq_len, gen_emb_dim=args.gen_emb_dim, mem_slots=args.mem_slots,
                                 head_size=args.head_size, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                                 start_token=args.start_token)

oracle_loader = RealDataLoader(args.batch_size, seq_len)

# placeholder definitions
x_real = tf.placeholder(tf.int32, [args.batch_size, seq_len], name="x_real")  # tokens of oracle sequences

temperature = tf.Variable(1., trainable=False, name='temperature')

x_fake_onehot_appr, x_fake, g_pretrain_loss, gen_o = generator(x_real=x_real, temperature=temperature)


with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    new_saver = tf.train.import_meta_graph(meta_file)
    new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    index_word_dict = get_oracle_file(data_file, oracle_file, seq_len)
    oracle_loader.create_batches(oracle_file)

    ckpt_name = 'test'
    gen_save_file = os.path.join(test_samples_dir, '{}.txt'.format(ckpt_name))
    generate_samples(sess, x_fake, args.batch_size, args.num_sentences, test_gen_file)
    get_real_test_file(test_gen_file, gen_save_file, index_word_dict)

    metrics = get_metrics(config, oracle_loader, test_file, gen_save_file, g_pretrain_loss, x_real, sess)
    metric_names = [metric.get_name() for metric in metrics]
    scores = [metric.get_score() for metric in metrics]
    msg = ckpt_name
    for (name, score) in zip(metric_names, scores):
        msg += ', ' + name + ': %.4f' % score
    print(msg)
