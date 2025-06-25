import sys
import numpy as np
import atexit

MPI = None

def _import_mpi(use_dill=False):
    try:
        from mpi4py import MPI as _MPI
        if use_dill:
            import dill
            _MPI.pickle.__init__(dill.dumps, dill.loads, dill.HIGHEST_PROTOCOL)
        return _MPI
    except ImportError:
        raise ImportError("Please install mpi4py")


class MPIPool:
    def __init__(self, comm=None, use_dill=False, evaluator_cls=None, transform_fn=None, evaluator_kwargs=None):
        global MPI
        if MPI is None:
            MPI = _import_mpi(use_dill=use_dill)

        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.master = 0
        self.rank = self.comm.Get_rank()
        self.evaluator_cls = evaluator_cls
        self.evaluator = None
        self.transform_fn = transform_fn
        self.evaluator_kwargs = evaluator_kwargs or {}


        atexit.register(lambda: MPIPool.close(self))

        if not self.is_master():
            if evaluator_cls is None:
                raise ValueError("Must provide evaluator_cls to construct evaluator on workers")
            self.evaluator = evaluator_cls(**self.evaluator_kwargs)
            self.wait()
            sys.exit(0)

        self.workers = set(range(self.comm.size))
        self.workers.discard(self.master)
        self.size = self.comm.Get_size() - 1

        if self.size == 0:
            raise ValueError("Need at least two MPI processes to run MPIPool")

    def is_master(self):
        return self.rank == 0

    def is_worker(self):
        return self.rank != 0

    def wait(self):
        if self.is_master():
            return

        status = MPI.Status()
        while True:
            task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)
            if task is None:
                break
            idx, arg = task
            result = self.evaluator(arg)
            self.comm.ssend((idx, result), dest=self.master, tag=status.tag)

    def map_array(self, xs: np.ndarray) -> np.ndarray:
        """
        Evaluate xs with shape (N, D) across workers, preserving input order.

        Parameters
        ----------
        xs : np.ndarray of shape (N, D)

        Returns
        -------
        np.ndarray of shape (N,)
        """
        if not self.is_master():
            self.wait()
            return

        if xs.ndim != 2:
            raise ValueError("Expected xs to have shape (N, D)")

        # Apply the transformation if desired.
        xp = self.transform_fn(xs) if self.transform_fn is not None else xs

        tasklist = list(enumerate(xp))
        resultlist = [None] * len(tasklist)
        workerset = self.workers.copy()
        pending = len(tasklist)

        while pending:
            if workerset and tasklist:
                worker = workerset.pop()
                task = tasklist.pop()
                self.comm.send(task, dest=worker, tag=task[0])

            if tasklist:
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag:
                    continue
            else:
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            idx, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            resultlist[idx] = result
            workerset.add(status.source)
            pending -= 1

        return np.array(resultlist)

    def close(self):
        if self.is_worker():
            return
        for worker in self.workers:
            self.comm.send(None, dest=worker, tag=0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
