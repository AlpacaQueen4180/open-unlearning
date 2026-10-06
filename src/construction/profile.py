"""Explicit construction batch profiles; independent of NPO normalization."""
import os

def world_size():
    value = int(os.environ.get('WORLD_SIZE', '1'))
    if value not in (1, 2):
        raise ValueError('Construction supports only accepted one/two rank profiles')
    return value

def accumulation(microbatch, world, global_batch=32):
    if world not in (1, 2) or microbatch < 1 or global_batch % (microbatch * world):
        raise ValueError('Global batch must be divisible by microbatch times world size')
    return global_batch // (microbatch * world)
