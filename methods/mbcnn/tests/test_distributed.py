from mbcnn.distributed import DistributedEvalSampler, reduce_max, reduce_sums


def test_distributed_eval_sampler_has_no_padding_or_duplicates() -> None:
    dataset = list(range(10))
    samplers = [
        DistributedEvalSampler(
            dataset,
            num_replicas=3,
            rank=rank,
            max_samples=8,
        )
        for rank in range(3)
    ]
    partitions = [list(sampler) for sampler in samplers]
    flattened = [index for partition in partitions for index in partition]

    assert sorted(flattened) == list(range(8))
    assert len(flattened) == len(set(flattened))
    assert [len(sampler) for sampler in samplers] == [3, 3, 2]


def test_reductions_are_identity_without_process_group() -> None:
    assert reduce_sums([1.0, 2.0], device="cpu") == [1.0, 2.0]
    assert reduce_max(3.0, device="cpu") == 3.0

