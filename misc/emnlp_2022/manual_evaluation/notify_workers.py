"""
aws configure
[access_key] AKIAR3Z6DKKOSVK2BIXW
[secret_access_key] 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
[region] us-east-1
"""
import boto3

workers = ['A110KENBXU7SUJ', 'A11S8IAAVDXCUS', 'A19R04IXWA4ZLX', 'A1DU5P3DMNJ44M', 'A1FP3SH704X01V', 'A1SCPXEIA8EYT5',
           'A1V2H0UF94ATWY', 'A20X14OMRL0YPZ', 'A22AKWWFAN7VQM', 'A27W025UEXS1G0', 'A2AAY4VT9L71SY', 'A2CK0OXMPOR9LE',
           'A31Z5TPD8QKE26', 'A320QA9HJFUOZO', 'A3C3Q963MQDPGT', 'A3OZ8KF0HWSVWK', 'A7R1OLLU1Z6RA', 'A7TUG10LNB586',
           'APGX2WZ59OWDN', 'ATR6RB1RULOC0', 'AXMPSUNKUBEIL']
workers_exclude = []
client = boto3.client('mturk', region_name='us-east-1')


def make_request(worker_id, subject, message):
    response = client.notify_workers(
        Subject=subject,
        MessageText=message,
        WorkerIds=[worker_id]
    )
    print(response)


if __name__ == '__main__':

    for n, worker in enumerate(workers):
        if worker in workers_exclude:
            continue
        print('Progress: [{}/{}]'.format(n + 1, len(workers)))
        body = 'Dear annotater {},\n\nThanks for working on our qualification HIT. We are delighted to announce that ' \
               'you are qualified to have an access to our main HITs. You did a great work in the qualification HIT ' \
               'and we hope you continue to work on our main HITs. Please find our HITs, where the task name start ' \
               'from  `QG Evaluation`. Let us know if you can not find or have no access to the HITs.\n\n' \
               'Best,\n' \
               'Asahi'.format(worker)
        make_request(
            worker_id=worker,
            subject='QG Evaluation: Announcement of the main batch to qualified workers.',
            message=body)
        # body = 'Dear qualified workers,\n\nThanks for working on our "QG Evaluation" task! ' \
        #        'We want to notify that we have just released two new batches now, and we appreciate if you can work on them. ' \
        #        'The task name is `QG Evaluation (main 2)`, ' \
        #        "so please search and find it on the worker's platform. Let us know if you can not find or have no access to the HITs.\n\n" \
        #        'Best,\n' \
        #        'Asahi'.format(worker)
        # make_request(
        #     worker_id=worker,
        #     subject='QG Evaluation: Notification of New Batches to Qualified Workers',
        #     message=body)
