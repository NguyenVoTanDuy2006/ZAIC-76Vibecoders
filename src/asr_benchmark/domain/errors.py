class BenchmarkError(RuntimeError):
    pass


class DatasetError(BenchmarkError):
    pass


class WorkerError(BenchmarkError):
    pass
