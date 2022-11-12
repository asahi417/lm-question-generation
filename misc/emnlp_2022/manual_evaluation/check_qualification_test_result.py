from glob import glob
import numpy as np
import pandas as pd

truth = {
    1: [[3], [3], [2]],
    2: [[3], [2, 3], [1]],
    3: [[2, 3], [1, 2], [1]],
    4: [[1, 2], [1], [1]],
}
blocked_worker = ['A70L26UXLTGLC']
files = glob('./mturk_output/qualification_test/*.csv')
df_result = pd.concat(pd.read_csv(i) for i in files)
df_result = df_result[df_result['AssignmentStatus'] != 'Approved']
worker_n = len(df_result)
result = {}
for i in range(worker_n):
    worker_id = df_result['WorkerId'].values[i]
    if worker_id in result:
        input('duplicated worker: {}'.format(worker_id))
    result[worker_id] = {
        n: [
            df_result['Answer.grammaticality_{}'.format(n)].values[i],
            df_result['Answer.understandability_{}'.format(n)].values[i],
            df_result['Answer.correctness_{}'.format(n)].values[i]
        ] for n in range(1, 6)}
inconsistent_workers = {k: v for k, v in result.items() if v[1] != v[5]}
print('inconsistent workers: {}'.format(len(inconsistent_workers)))
result = {k: v for k, v in result.items() if v[1] == v[5] and k not in blocked_worker}
print('total workers: {}'.format(len(result)))

errors = []
errors_q_wise = {}
for k, v in result.items():
    for job_id in range(1, 5):
        tmp_v = v[job_id]
        true = truth[job_id]
        e = [int(a not in b) for a, b in zip(tmp_v, true)]
        errors.append(e)
        if job_id not in errors_q_wise:
            errors_q_wise[job_id] = [e]
        else:
            errors_q_wise[job_id].append(e)
errors = np.array(errors)
total_n = len(errors)
if total_n != 0:
    print('\nError (total)')
    print('\t grammaticality: {}/{} ({})'.format(errors.sum(0)[0], total_n, errors.sum(0)[0]/total_n))
    print('\t understandabil: {}/{} ({})'.format(errors.sum(0)[1], total_n, errors.sum(0)[1]/total_n))
    print('\t correctness   : {}/{} ({})'.format(errors.sum(0)[2], total_n, errors.sum(0)[2]/total_n))
    print('\nError (question-wise)')
    for i in range(1, 5):
        print('Question {} ({})'.format(i, truth[i]))
        errors = np.array(errors_q_wise[i])
        total_n = len(errors_q_wise[i])
        print('\t grammaticality: {}/{} ({})'.format(errors.sum(0)[0], total_n, errors.sum(0)[0]/total_n))
        print('\t understandabil: {}/{} ({})'.format(errors.sum(0)[1], total_n, errors.sum(0)[1]/total_n))
        print('\t correctness   : {}/{} ({})'.format(errors.sum(0)[2], total_n, errors.sum(0)[2]/total_n))


# qualification
print()
acceptance_threshold = 2
print('Worker Selection: acceptance threshold ({})'.format(acceptance_threshold))
accepted_workers = []
declined_workers = []
for k, v in result.items():
    score = 0
    flag = True
    # sanity check: answer consistency by comparing 1 with 5 (they are same).
    for _k, _v in list(v.items())[:-1]:
        diff = [int(a not in b) for a, b in zip(_v, truth[_k])]
        score += sum(diff)
    if score > acceptance_threshold:
        flag = False
    # score = score / 4
    if flag:
        print('passed worker: {}, average mistake: {}'.format(k, score))
        accepted_workers.append([k, score])
    else:
        declined_workers.append([k, score])
accepted_workers = dict(accepted_workers)
declined_workers = dict(declined_workers)
print('declined workers: {}'.format(len(declined_workers)))
print('accepted workers: {}'.format(len(accepted_workers)))
print(list(accepted_workers.keys()))

# Assign Qualification
user_file = glob('./mturk_output/users/*.csv')
if len(user_file) == 0:
    exit()
if len(user_file) > 1:
    raise ValueError('found more than one user file: {}'.format(user_file))
user_file = user_file[0]
print('Overwrite userfile: {}'.format(user_file))
df = pd.read_csv(user_file)
print('- current qualified user: {}'.format(len(df[df['CURRENT-QG Evaluation Qualified'] == 100])))
df = df[df['CURRENT BlockStatus'] != 'Blocked']
df = df[df['CURRENT-QG Evaluation Qualified'] != 100]
df = df[df['CURRENT-QG Evaluation Qualified'] != 0]
tmp = df['UPDATE-QG Evaluation Qualified'].fillna(0).astype(int)

tmp[[k in accepted_workers.keys() for k in df['Worker ID']]] = 100
if len([k for k in accepted_workers.keys() if k in df['Worker ID'].values]) != len(accepted_workers):
    print('WARNING: drop users {} -> {}'.format(
        len([k for k in accepted_workers.keys() if k in df['Worker ID']]), len(accepted_workers)))
df['UPDATE-QG Evaluation Qualified'] = tmp
df['UPDATE-QG Evaluation Qualified (new worker)'] = 100
print('- new qualified user    : {}'.format(len(df[df['UPDATE-QG Evaluation Qualified'] == 100])))
df.to_csv(user_file.replace('.csv', '.update.csv'), index=False)
