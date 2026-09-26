'''
The pipeline service: the headless library pipeline as a long-running process, driven over HTTP and, optionally, on a
schedule (design/pipeline-service.md). Qt-free, like the rest of `pipeline/`.

    config.py    ServiceConfig: service.yaml, and the secrets the environment gives
    context.py   a job's profile, read again for every job, with the environment's secrets applied
    jobs.py      JobManager: one job at a time, a queue, cancel, progress, events and a history kept on disk
    work.py      what each kind of job does: scan, run (a selection through a stage) and bulk accept
'''
