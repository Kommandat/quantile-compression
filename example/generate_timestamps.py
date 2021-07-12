import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq
import random
from datetime import date, datetime, timedelta, timezone
import os

n = 10 ** 3

def random_dates(n):
  """
    Returns `n` random dates between `1900-01-01` & current timestamp
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

arr = random_dates(n)

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
  for num in ints96:
    curr = num
    for _ in range(12):
      byte = curr & 255 
      curr = curr >> 8
      byte_representation.append(byte)
    
  with open(f'data/txt/i96_{name}.txt', 'w') as f:
    f.write(joined)
  with open(f'data/binary/i96_{name}.bin', 'wb') as f:
    f.write(bytes(byte_representation))

write_i96(random_dates(n), 'timestamps')