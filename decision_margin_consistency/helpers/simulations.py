import numpy as np
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def run_simulation(p1, p2, num_trials, repetitions):
    samples1 = gen_samples(num_trials, p1, repetitions)
    samples2 = gen_samples(num_trials, p2, repetitions)
    # estimate accuracy
    p1Hat = samples1.mean(axis=0);
    p2Hat = samples2.mean(axis=0);
    # calculate cexp and cobs
    cexpCol= p1Hat*p2Hat + (1-p1Hat)*(1-p2Hat)
    cobsCol= (samples1 == samples2).mean(axis=0)

    return cexpCol, cobsCol