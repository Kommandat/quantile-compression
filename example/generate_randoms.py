# python 3
# pip requirement: numpy, pyarrow

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq
import random
from datetime import datetime, timedelta, timezone
import os

n = 10 ** 6

os.makedirs('data/txt', exist_ok=True)
os.makedirs('data/parquet', exist_ok=True)
os.makedirs('data/snappy_parquet', exist_ok=True)
os.makedirs('data/gzip_parquet', exist_ok=True)
os.makedirs('data/binary', exist_ok=True)

def write_i64(arr, name):
  if arr.dtype != np.int64:
    floored = np.floor(arr).astype(np.int64)
  else:
    floored = arr
  ints = [str(x) for x in floored]
  joined = '\n'.join(ints)
  with open(f'data/txt/i64_{name}.txt', 'w') as f:
    f.write(joined)
  with open(f'data/binary/i64_{name}.bin', 'wb') as f:
    f.write(floored.tobytes())
  table = pa.Table.from_pydict({'nums': floored})
  pq.write_table(table, f'data/parquet/i64_{name}.parquet', compression='NONE')
  pq.write_table(table, f'data/snappy_parquet/i64_{name}.snappy.parquet', compression='snappy')

def int96(num):
  if num >= 0:
    return num
  else:
    return 2**96 + num
  
def bin96(num):
  if num >= 0:
    return '{:096b}'.format(num)
  else:
    return '{:096b}'.format(2**96 + num)

def write_i96(arr, name):
  ints96 = (int96(x) for x in arr)
  bins96 = (bin96(x) for x in arr)

  joined = '\n'.join(bins96)

  byte_representation = []
  bytes_stack = []
  for num in ints96:
    curr = num
    for _ in range(12):
      byte = curr & 255
      curr = curr >> 8
      bytes_stack.append(byte)
    
    # We use a stack to add bytes to the byte representation
    # because we want the most significant byte first (i.e. the
    # last byte added to the stack)
    while bytes_stack:
        byte_representation.append(bytes_stack.pop())

  with open(f'data/txt/i96_{name}.txt', 'w') as f:
    f.write(joined)
  with open(f'data/binary/i96_{name}.bin', 'wb') as f:
    f.write(bytes(byte_representation))

def write_f64(arr, name):
  arr = arr.astype(np.float64)
  floats = [str(x) for x in arr]
  joined = '\n'.join(floats)
  with open(f'data/txt/f64_{name}.txt', 'w') as f:
    f.write(joined)
  with open(f'data/binary/f64_{name}.bin', 'wb') as f:
    f.write(arr.tobytes())
  table = pa.Table.from_pydict({'nums': arr})
  pq.write_table(table, f'data/parquet/f64_{name}.parquet', compression='NONE')
  pq.write_table(table, f'data/snappy_parquet/f64_{name}.snappy.parquet', compression='snappy')

write_i64(np.random.normal(scale=1.0, size=n), 'normal1')
write_i64(np.random.normal(scale=10.0, size=n), 'normal10')
write_i64(np.random.normal(scale=1000000.0, size=n), 'normal1M')

write_i64(np.random.geometric(p=0.5, size=n), 'geo2')
write_i64(np.random.geometric(p=0.000001, size=n), 'geo1M')

def fixed_median_lomax(a, median):
  unscaled_median = 2 ** (1 / a) - 1
  return np.random.pareto(a=a, size=n) / unscaled_median * median
write_i64(fixed_median_lomax(0.5, 1000), 'lomax05')
write_i64(fixed_median_lomax(1.5, 1000), 'lomax15')
write_i64(fixed_median_lomax(2.5, 1000), 'lomax25')

write_i64(np.random.randint(-2**63, 2**63, size=n), 'uniform')

write_i64(np.repeat(77777, n), 'constant')

write_i64(np.where(np.random.uniform(size=n) < 0.5, -2**63, 2**63 - 1), 'extremes')

write_i64(np.random.binomial(1, p=0.01, size=n), 'sparse')

dollars = np.floor(fixed_median_lomax(1.5, 5)).astype(np.int64)
cents = np.random.randint(0, 100, size=n)
p = np.random.uniform(size=n)
cents[p < 0.9] = 99
cents[p < 0.75] = 98
cents[p < 0.6] = 95
cents[p < 0.45] = 75
cents[p < 0.4] = 50
cents[p < 0.25] = 25
cents[p < 0.15] = 0
total_cents = dollars * 100 + cents
write_i64(dollars, 'dollars')
write_i64(cents, 'cents')
write_i64(total_cents, 'total_cents')

write_f64(np.random.normal(size=n), 'normal_at_0')
write_f64(np.random.normal(loc=1000.0, size=n), 'normal_at_1000')

edge_case_floats = np.random.normal(size=n)
p = np.random.uniform(size=n)
edge_case_floats[p < 0.5] *= 2.0 ** -1022 # often denormalized values
edge_case_floats[p < 0.4] = np.inf
edge_case_floats[p < 0.3] = np.nan
edge_case_floats[p < 0.2] = -np.nan  # yes, it is different
edge_case_floats[p < 0.1] = np.NINF
write_f64(edge_case_floats, 'edge_cases')

def random_dates(n):
  """
  Returns `n` random dates between `1900-01-01` & current timestamp
  as nanoseconds from 1970-01-01
  """
  start = datetime(1900, 1, 1, tzinfo=timezone.utc)
  end = datetime.now(timezone.utc)
  epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

  out = []
  for _ in range(n):
    random_date = start + (end - start) * random.random()
    nanosecond_epoch = int(1E3 * (random_date - epoch) / timedelta(microseconds=1))
    nanosecond_epoch += int(1000 * random.random()) # Add a random number of nanoseconds
    out.append(nanosecond_epoch)
  return out

write_i96(random_dates(n), 'timestamps')
