use std::cmp::Ordering;
use std::convert::TryInto;

use crate::compressor::Compressor;
use crate::decompressor::Decompressor;
use crate::types::{DataType, NumberLike};

impl NumberLike for u32 {
  fn num_eq(&self, other: &Self) -> bool {
    self.eq(other)
  }

  fn num_cmp(&self, other: &Self) -> Ordering {
    self.cmp(other)
  }
}

pub struct U32DataType {}

impl DataType<u32> for U32DataType {
  const HEADER_BYTE: u8 = 4;
  const BIT_SIZE: usize = 32;

  fn offset_diff(upper: u32, lower: u32) -> u64 {
    (upper - lower) as u64
  }

  fn add_offset(lower: u32, off: u64) -> u32 {
    lower + off as u32
  }

  fn bytes_from(num: u32) -> Vec<u8> {
    num.to_be_bytes().to_vec()
  }

  fn from_bytes(bytes: Vec<u8>) -> u32 {
    u32::from_be_bytes(bytes.try_into().unwrap())
  }
}

pub type U32Compressor = Compressor<u32, U32DataType>;
pub type U32Decompressor = Decompressor<u32, U32DataType>;
