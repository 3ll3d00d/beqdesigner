'''
Making jobs over the service's HTTP interface in tests (design/jobs-and-executions.md §4): POST /v1/jobs, then follow
the job's first execution to its end.
'''
import time


def make_job(client, headers, kind, request=None, **fields):
    ''' POST /v1/jobs: a one-off `kind` job (or, with `every`, a recurring one) asked to do `request`. '''
    return client.post('/v1/jobs', json={'kind': kind, 'request': request or {}, **fields}, headers=headers)


def first_execution(client, headers, response, timeout=30.0):
    ''' The first execution of the job `response` made, once it has finished. '''
    assert response.status_code == 201, response.text
    job_id = response.json()['id']
    deadline = time.time() + timeout
    while True:
        job = client.get(f'/v1/jobs/{job_id}', headers=headers).json()
        if job['last_execution']:
            execution = client.get(f"/v1/executions/{job['last_execution']}", headers=headers).json()
            if execution['state'] not in ('queued', 'running'):
                return execution
        assert time.time() < deadline, f'job {job_id} did not finish: {job}'
        time.sleep(0.01)
