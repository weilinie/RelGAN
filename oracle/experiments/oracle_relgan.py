import os
from subprocess import call
import sys, time

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = '0'
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    print('Missing argument: job_id and gpu_id.')
    quit()

# Executables
executable = 'python3'

# Arguments
architecture = ['rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'lstm_vanilla', 'lstm_vanilla', 'lstm_vanilla', 'lstm_vanilla']
gantype =      ['RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN']
opt_type =     ['adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam']
temperature =  ['2', '5', '10', '100', '2', '5', '10', '100']
d_lr =         ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']
gadv_lr =      ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']
mem_slots =    ['1', '1', '1', '1', '1', '1', '1', '1']
head_size =    ['256', '256', '256', '256', '256', '256', '256', '256']
num_heads =    ['2', '2', '2', '2', '2', '2', '2', '2']

bs = '64'
seed = '124'
gpre_lr = '1e-2'
hidden_dim = '32'
seq_len = '20'
dataset = 'oracle'

gsteps = '1'
dsteps = '5'
gen_emb_dim = '32'
dis_emb_dim = '64'
num_rep = '64'
sn = False
decay = False
adapt = 'exp'
npre_epochs = '150'
nadv_steps = '3000'
ntest = '20'

# Paths
rootdir = '../..'
scriptname = 'run.py'
cwd = os.path.dirname(os.path.abspath(__file__))

outdir = os.path.join(cwd, 'out', time.strftime("%Y%m%d"), dataset,
                      'oracle_{}_{}_{}_bs{}_sl{}_sn{}_dec{}_ad-{}_npre{}_nadv{}_ms{}_hs{}_nh{}_ds{}_dlr{}_glr{}_tem{}_demb{}_nrep{}_hdim{}_sd{}'.
                      format(architecture[job_id], gantype[job_id], opt_type[job_id], bs, seq_len, int(sn),
                             int(decay), adapt, npre_epochs, nadv_steps, mem_slots[job_id], head_size[job_id],
                             num_heads[job_id], dsteps, d_lr[job_id], gadv_lr[job_id], temperature[job_id],
                             dis_emb_dim, num_rep, hidden_dim, seed))

args = [
    # Architecture
    '--gf-dim', '64',
    '--df-dim', '64',
    '--g-architecture', architecture[job_id],
    '--d-architecture', architecture[job_id],
    '--gan-type', gantype[job_id],
    '--hidden-dim', hidden_dim,

    # Training
    '--gsteps', gsteps,
    '--dsteps', dsteps,
    '--npre-epochs', npre_epochs,
    '--nadv-steps', nadv_steps,
    '--ntest', ntest,
    '--d-lr', d_lr[job_id],
    '--gpre-lr', gpre_lr,
    '--gadv-lr', gadv_lr[job_id],
    '--batch-size', bs,
    '--log-dir', os.path.join(outdir, 'tf_logs'),
    '--sample-dir', os.path.join(outdir, 'samples'),
    '--optimizer', opt_type[job_id],
    '--seed', seed,
    '--temperature', temperature[job_id],
    '--adapt', adapt,

    # evaluation
    '--nll-oracle',
    '--nll-gen',
    # '--doc-embsim',

    # relational memory
    '--mem-slots', mem_slots[job_id],
    '--head-size', head_size[job_id],
    '--num-heads', num_heads[job_id],

    # dataset
    '--dataset', dataset,
    '--vocab-size', '5000',
    '--start-token', '0',
    '--seq-len', seq_len,
    '--num-sentences', '10000',
    '--gen-emb-dim', gen_emb_dim,
    '--dis-emb-dim', dis_emb_dim,
    '--num-rep', num_rep,
    '--data-dir', './data']

if sn:
    args += ['--sn']
if decay:
    args += ['--decay']


# Run
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
