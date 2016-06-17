use libc::{c_int, c_uint};
use std::ptr::{copy, null};
use std::slice::from_raw_parts;
use std::cmp::{max, min};
use {mz_adler32, SizeOf};
use tdefl::memory_specific_constants::*;

use tdefl::tdefl_status::*;
use tdefl::tdefl_flush::*;

// ------------------- Low-level Compression API Definitions

// Set TDEFL_LESS_MEMORY to 1 to use less memory (compression will be slightly slower, and raw/dynamic blocks will be output more frequently).
const TDEFL_LESS_MEMORY: isize = 0;

/// tdefl_init() compression flags logically OR'd together (low 12 bits contain the max. number of probes per dictionary search):
/// TDEFL_DEFAULT_MAX_PROBES: The compressor defaults to 128 dictionary probes per dictionary search. 0=Huffman only, 1=Huffman+LZ (fastest/crap compression), 4095=Huffman+LZ (slowest/best compression).
///
/// TDEFL_WRITE_ZLIB_HEADER: If set, the compressor outputs a zlib header before the deflate data, and the Adler-32 of the source data at the end. Otherwise, you'll get raw deflate data.
/// TDEFL_COMPUTE_ADLER32: Always compute the adler-32 of the input data (even when not writing zlib headers).
/// TDEFL_GREEDY_PARSING_FLAG: Set to use faster greedy parsing, instead of more efficient lazy parsing.
/// TDEFL_NONDETERMINISTIC_PARSING_FLAG: Enable to decrease the compressor's initialization time to the minimum, but the output may vary from run to run given the same input (depending on the contents of memory).
/// TDEFL_RLE_MATCHES: Only look for RLE matches (matches with a distance of 1)
/// TDEFL_FILTER_MATCHES: Discards matches <= 5 chars if enabled.
/// TDEFL_FORCE_ALL_STATIC_BLOCKS: Disable usage of optimized Huffman tables.
/// TDEFL_FORCE_ALL_RAW_BLOCKS: Only use raw (uncompressed) deflate blocks.
bitflags! {
  pub flags CompressionFlags: u32
  {
    const TDEFL_HUFFMAN_ONLY                  =       0,
    const TDEFL_FASTEST_MAX_PROBES            =     0x1,
    const TDEFL_DEFAULT_MAX_PROBES            =    0x80,
    const TDEFL_MAX_PROBES_MASK               =   0xFFF,
    const TDEFL_WRITE_ZLIB_HEADER             = 0x01000,
    const TDEFL_COMPUTE_ADLER32               = 0x02000,
    const TDEFL_GREEDY_PARSING_FLAG           = 0x04000,
    const TDEFL_NONDETERMINISTIC_PARSING_FLAG = 0x08000,
    const TDEFL_RLE_MATCHES                   = 0x10000,
    const TDEFL_FILTER_MATCHES                = 0x20000,
    const TDEFL_FORCE_ALL_STATIC_BLOCKS       = 0x40000,
    const TDEFL_FORCE_ALL_RAW_BLOCKS          = 0x80000
  }
}

/// Output stream interface. The compressor uses this interface to write compressed data. It'll typically be called TDEFL_OUT_BUF_SIZE at a time.
type tdefl_put_buf_func_ptr<'a> = &'a mut FnMut<(*const u8, usize), Output = bool>;

const TDEFL_MAX_HUFF_TABLES: usize = 3;
const TDEFL_MAX_HUFF_SYMBOLS_0: usize = 288;
const TDEFL_MAX_HUFF_SYMBOLS_1: usize = 32;
const TDEFL_MAX_HUFF_SYMBOLS_2: usize = 19;
const TDEFL_MIN_MATCH_LEN: usize = 3;
const TDEFL_MAX_MATCH_LEN: usize = 258;
const TDEFL_LZ_DICT_SIZE: usize = 32768;
const TDEFL_LZ_DICT_SIZE_MASK: usize = (TDEFL_LZ_DICT_SIZE - 1);

// TDEFL_OUT_BUF_SIZE MUST be large enough to hold a single entire compressed output block (using static/fixed Huffman codes).
#[cfg(TDEFL_LESS_MEMORY)]
mod memory_specific_constants {
  pub const TDEFL_LZ_CODE_BUF_SIZE: usize = 24 * 1024;
  pub const TDEFL_OUT_BUF_SIZE: usize = (TDEFL_LZ_CODE_BUF_SIZE * 13 ) / 10;
  pub const TDEFL_MAX_HUFF_SYMBOLS: usize = 288;
  pub const TDEFL_LZ_HASH_BITS: usize = 12;
  pub const TDEFL_LEVEL1_HASH_SIZE_MASK: usize = 4095;
  pub const TDEFL_LZ_HASH_SHIFT: usize = (TDEFL_LZ_HASH_BITS + 2) / 3;
  pub const TDEFL_LZ_HASH_SIZE: usize = 1 << TDEFL_LZ_HASH_BITS;
}
#[cfg(not(TDEFL_LESS_MEMORY))]
mod memory_specific_constants {
  pub const TDEFL_LZ_CODE_BUF_SIZE: usize = 64 * 1024;
  pub const TDEFL_OUT_BUF_SIZE: usize = (TDEFL_LZ_CODE_BUF_SIZE * 13 ) / 10;
  pub const TDEFL_MAX_HUFF_SYMBOLS: usize = 288;
  pub const TDEFL_LZ_HASH_BITS: usize = 15;
  pub const TDEFL_LEVEL1_HASH_SIZE_MASK: usize = 4095;
  pub const TDEFL_LZ_HASH_SHIFT: usize = (TDEFL_LZ_HASH_BITS + 2) / 3;
  pub const TDEFL_LZ_HASH_SIZE: usize = 1 << TDEFL_LZ_HASH_BITS;
}

// The low-level tdefl functions below may be used directly if the above helper functions aren't flexible enough. The low-level functions don't make any heap allocations, unlike the above helper functions.
#[derive(PartialEq, Clone, Copy)]
#[repr(i8)]
enum tdefl_status
{
  TDEFL_STATUS_BAD_PARAM = -2,
  TDEFL_STATUS_PUT_BUF_FAILED = -1,
  TDEFL_STATUS_OKAY = 0,
  TDEFL_STATUS_DONE = 1,
}

/// Must map to MZ_NO_FLUSH, MZ_SYNC_FLUSH, etc. enums
#[derive(PartialEq, Clone, Copy)]
enum tdefl_flush
{
  TDEFL_NO_FLUSH = 0,
  TDEFL_SYNC_FLUSH = 2,
  TDEFL_FULL_FLUSH = 3,
  TDEFL_FINISH = 4
}

/// tdefl's compression state structure.
struct tdefl_compressor <'a>
{
  m_pPut_buf_func: Option<tdefl_put_buf_func_ptr<'a>>,
  // m_pPut_buf_user: *mut c_void,
  m_flags: CompressionFlags, m_max_probes: [usize; 2],
  m_greedy_parsing: bool,
  m_adler32: u32, m_lookahead_pos: usize, m_lookahead_size: usize, m_dict_size: usize,
  m_pLZ_code_buf: *mut u8, m_pLZ_flags: *mut u8, m_pOutput_buf: *mut u8, m_pOutput_buf_end: *mut u8,
  m_num_flags_left: usize, m_total_lz_bytes: usize, m_lz_code_buf_dict_pos: usize, m_bits_in: usize, m_bit_buffer: u64,
  m_saved_match_dist: usize, m_saved_match_len: usize, m_saved_lit: u8, m_output_flush_ofs: usize, m_output_flush_remaining: usize, m_finished: bool, m_block_index: usize, m_wants_to_finish: bool,
  m_prev_return_status: tdefl_status,
  m_pIn_buf: *const u8,
  m_pOut_buf: *mut u8,
  m_pIn_buf_size: *mut usize, m_pOut_buf_size: *mut usize,
  m_flush: tdefl_flush,
  m_pSrc: *const u8,
  m_src_buf_left: usize, m_out_buf_ofs: usize,
  m_dict: [u8; TDEFL_LZ_DICT_SIZE + TDEFL_MAX_MATCH_LEN - 1],
  // TODO verify this is the right order of indexes
  m_huff_count: [[u16; TDEFL_MAX_HUFF_SYMBOLS]; TDEFL_MAX_HUFF_TABLES],
  m_huff_codes: [[u16; TDEFL_MAX_HUFF_SYMBOLS]; TDEFL_MAX_HUFF_TABLES],
  m_huff_code_sizes: [[u8; TDEFL_MAX_HUFF_SYMBOLS]; TDEFL_MAX_HUFF_TABLES],
  m_lz_code_buf: [u8; TDEFL_LZ_CODE_BUF_SIZE],
  m_next: [u16; TDEFL_LZ_DICT_SIZE],
  m_hash: [u16; TDEFL_LZ_HASH_SIZE],
  m_output_buf: [u8; TDEFL_OUT_BUF_SIZE],
}

/// Initializes the compressor.
/// There is no corresponding deinit() function because the tdefl API's do not dynamically allocate memory.
/// pBut_buf_func: If NULL, output data will be supplied to the specified callback. In this case, the user should call the tdefl_compress_buffer() API for compression.
/// If pBut_buf_func is NULL the user should always call the tdefl_compress() API.
/// flags: See the above enums (TDEFL_HUFFMAN_ONLY, TDEFL_WRITE_ZLIB_HEADER, etc.)
impl<'a> tdefl_compressor<'a> {
  unsafe fn new<'b: 'a>(pPut_buf_func: tdefl_put_buf_func_ptr<'b>, flags: CompressionFlags) -> tdefl_compressor<'a>
  {
    // if !flags.contains(TDEFL_NONDETERMINISTIC_PARSING_FLAG) {for i in d.m_hash.iter_mut() {*i = 0},}
    let mut d = tdefl_compressor {
      m_pPut_buf_func: Some(pPut_buf_func),
      m_flags: flags,
      m_max_probes: [1 + ((flags.bits() as usize & 0xFFF) + 2) / 3, 1 + (((flags.bits() as usize & 0xFFF) >> 2) + 2) / 3],
      m_greedy_parsing: flags.contains(TDEFL_GREEDY_PARSING_FLAG),
      m_lookahead_pos: 0,
      m_lookahead_size: 0,
      m_dict_size: 0,
      m_total_lz_bytes: 0,
      m_lz_code_buf_dict_pos: 0,
      m_bits_in: 0,
      m_output_flush_ofs: 0,
      m_output_flush_remaining: 0,
      m_block_index: 0,
      m_bit_buffer: 0,
      m_finished: false, m_wants_to_finish: false,
      m_pLZ_code_buf: null::<u8>() as *mut u8, m_pLZ_flags: null::<u8>() as *mut u8, m_num_flags_left: 8,
      m_pOutput_buf: null::<u8>() as *mut u8, m_pOutput_buf_end: null::<u8>() as *mut u8, m_prev_return_status: TDEFL_STATUS_OKAY,
      m_saved_match_dist: 0, m_saved_match_len: 0, m_saved_lit: 0, m_adler32: 1,
      m_pIn_buf: null::<u8>(), m_pOut_buf: null::<u8>() as *mut u8,
      m_pIn_buf_size: null::<usize>() as *mut usize, m_pOut_buf_size: null::<usize>() as *mut usize,
      m_flush: TDEFL_NO_FLUSH, m_pSrc: null(), m_src_buf_left: 0, m_out_buf_ofs: 0,
      m_dict: [0u8; TDEFL_LZ_DICT_SIZE + TDEFL_MAX_MATCH_LEN - 1],
      m_huff_count: [[0u16; TDEFL_MAX_HUFF_SYMBOLS]; TDEFL_MAX_HUFF_TABLES],
      m_huff_codes: [[0u16; TDEFL_MAX_HUFF_SYMBOLS]; TDEFL_MAX_HUFF_TABLES],
      m_huff_code_sizes: [[0u8; TDEFL_MAX_HUFF_SYMBOLS]; TDEFL_MAX_HUFF_TABLES],
      m_lz_code_buf: [0u8; TDEFL_LZ_CODE_BUF_SIZE],
      m_next: [0u16; TDEFL_LZ_DICT_SIZE],
      m_hash: [0u16; TDEFL_LZ_HASH_SIZE],
      m_output_buf: [0u8; TDEFL_OUT_BUF_SIZE],
    };
    d.m_pLZ_code_buf = d.m_lz_code_buf.as_mut_ptr().offset(1); d.m_pLZ_flags = d.m_lz_code_buf.as_mut_ptr();
    d.m_pOutput_buf = d.m_output_buf.as_mut_ptr(); d.m_pOutput_buf_end = d.m_output_buf.as_mut_ptr();
    return d;
  }
}

fn tdefl_get_prev_return_status(d: &mut tdefl_compressor) -> tdefl_status
{
  return d.m_prev_return_status;
}

fn tdefl_get_adler32(d: &mut tdefl_compressor) -> u32
{
  return d.m_adler32;
}

// ------------------- Low-level Compression (independent from all decompression API's)

// Purposely making these tables static for faster init and thread safety.
const S_TDEFL_LEN_SYM: [u16; 256] = [
  257,258,259,260,261,262,263,264,265,265,266,266,267,267,268,268,269,269,269,269,270,270,270,270,271,271,271,271,272,272,272,272,
  273,273,273,273,273,273,273,273,274,274,274,274,274,274,274,274,275,275,275,275,275,275,275,275,276,276,276,276,276,276,276,276,
  277,277,277,277,277,277,277,277,277,277,277,277,277,277,277,277,278,278,278,278,278,278,278,278,278,278,278,278,278,278,278,278,
  279,279,279,279,279,279,279,279,279,279,279,279,279,279,279,279,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,
  281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,
  282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
  283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,
  284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,285 ];

const S_TDEFL_LEN_EXTRA: [u8; 256] = [
  0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0 ];

const S_TDEFL_SMALL_DIST_SYM: [u8; 512] = [
  0,1,2,3,4,4,5,5,6,6,6,6,7,7,7,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,
  11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,
  13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,
  14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
  14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
  15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16,16,16,16,
  16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
  16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
  16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,17,17,17,17,17,
  17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,
  17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,
  17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17 ];

const S_TDEFL_SMALL_DIST_EXTRA: [u8; 512] = [
  0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7 ];

const S_TDEFL_LARGE_DIST_SYM: [u8; 128] = [
  0,0,18,19,20,20,21,21,22,22,22,22,23,23,23,23,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,26,26,26,
  26,26,26,26,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,
  28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29 ];

const S_TDEFL_LARGE_DIST_EXTRA: [u8; 128] = [
  0,0,8,8,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
  12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
  13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13 ];

//#[repr(packed)]
#[derive(Clone, Copy)]
struct tdefl_sym_freq { m_key: u16, m_sym_index: u16 }
impl tdefl_sym_freq { fn new() -> tdefl_sym_freq { tdefl_sym_freq {m_key: 0, m_sym_index: 0} } }
/// Radix sorts tdefl_sym_freq[] array by 16-bit key m_key. Returns ptr to sorted values.
fn tdefl_radix_sort_syms <'a>(num_syms: usize, pSyms0: &'a mut [tdefl_sym_freq], pSyms1: &'a mut [tdefl_sym_freq]) -> &'a mut [tdefl_sym_freq]
{
  let mut total_passes: usize = 2; let mut i: usize = 0; let mut hist = [0usize; 256 * 2];
  let mut pCur_syms: &mut [tdefl_sym_freq] = pSyms0; let mut pNew_syms: &mut [tdefl_sym_freq] = pSyms1;
  while i < num_syms { let freq: usize = pCur_syms[i].m_key as usize; hist[freq & 0xFF]+=1; hist[256 + ((freq >> 8) & 0xFF)]+=1; i+=1}
  while (total_passes > 1) && (num_syms == hist[(total_passes - 1) * 256]) {total_passes-=1};

  let mut pass_shift: usize = 0; let mut pass: usize = 0;
  while pass < total_passes
  {
    let pHist: &[usize] = &hist[pass << 8 .. 256];
    let mut offsets = [0usize; 256]; let mut cur_ofs: usize = 0;
    i = 0;
    while i < 256 { offsets[i] = cur_ofs; cur_ofs += pHist[i]; i+=1 }
    i = 0;
    while i < num_syms {
      let temp_index = (pCur_syms[i].m_key >> pass_shift) as usize & 0xFF;
      pNew_syms[offsets[temp_index]] = pCur_syms[i];
      offsets[temp_index] += 1;
      i+=1;
    }
    //{ let t = pCur_syms; pCur_syms = pNew_syms; pNew_syms = t; }
    ::std::mem::swap(&mut pCur_syms, &mut pNew_syms);
    pass += 1; pass_shift += 8;
  }
  return pCur_syms;
}

/// tdefl_calculate_minimum_redundancy() originally written by: Alistair Moffat, alistair@cs.mu.oz.au, Jyrki Katajainen, jyrki@diku.dk, November 1996.
fn tdefl_calculate_minimum_redundancy(A: &mut [tdefl_sym_freq], n: usize)
{
  let mut root: usize = 0; let mut leaf: usize = 2; let mut next: usize = 1; let mut avbl: usize = 1; let mut used: usize = 0; let mut dpth: c_int = 0;
  if n==0 {return;} else {if n==1 { A[0].m_key = 1; return; }}
  A[0].m_key += A[1].m_key;
  while next < n-1
  {
    if leaf>=n || A[root].m_key<A[leaf].m_key { A[next].m_key = A[root].m_key; A[root].m_key = next as u16; root+=1; } else {A[next].m_key = A[leaf].m_key; leaf+=1;}
    if leaf>=n || (root<next && A[root].m_key<A[leaf].m_key) { A[next].m_key = (A[next].m_key + A[root].m_key) as u16; A[root].m_key = next as u16; root+=1 }
    else {A[next].m_key = (A[next].m_key + A[leaf].m_key) as u16; leaf+=1;}
    next+=1;
  }
  A[n-2].m_key = 0; next=n-3; while next>=0 {A[next].m_key = A[A[next].m_key as usize].m_key+1; next-=1;}
  root = n-2; next = n-1;
  while avbl>0
  {
    while root>=0 && A[root].m_key as c_int == dpth { used+=1; root-=1; }
    while avbl>used { A[next].m_key = dpth as u16; next-=1; avbl-=1; }
    avbl = 2*used; dpth+=1; used = 0;
  }
}

/// Limits canonical Huffman code table's max code size.
const TDEFL_MAX_SUPPORTED_HUFF_CODESIZE: usize = 32;
fn tdefl_huffman_enforce_max_code_size(pNum_codes: &mut[usize], code_list_len: usize, max_code_size: usize)
{
  let mut i: usize; let mut total: u32 = 0; if code_list_len <= 1 {return;}
  i = max_code_size + 1;
  while i <= TDEFL_MAX_SUPPORTED_HUFF_CODESIZE {pNum_codes[max_code_size] += pNum_codes[i]; i+=1;}
  i = max_code_size;
  while i > 0 {total += (pNum_codes[i] as u32) << (max_code_size - i); i-=1;}
  while total != (1 << max_code_size)
  {
    pNum_codes[max_code_size]-=1;
    i = max_code_size - 1;
    while i > 0 {if pNum_codes[i] != 0 { pNum_codes[i]-=1; pNum_codes[i + 1] += 2; break; }; i-=1;}
    total-=1;
  }
}

fn tdefl_optimize_huffman_table(d: &mut tdefl_compressor, table_num: usize, table_len: usize, code_size_limit: usize, static_table: bool)
{
  let mut i: usize = 0; let mut num_codes= [0 as usize; 1 + TDEFL_MAX_SUPPORTED_HUFF_CODESIZE]; let mut next_code= [0 as usize; TDEFL_MAX_SUPPORTED_HUFF_CODESIZE + 1];
  if static_table
  {
    while i < table_len { num_codes[d.m_huff_code_sizes[table_num][i] as usize]+=1; i+=1 }
  }
  else
  {
    let mut syms0 = [tdefl_sym_freq::new(); TDEFL_MAX_HUFF_SYMBOLS]; let mut syms1 = [tdefl_sym_freq::new(); TDEFL_MAX_HUFF_SYMBOLS]; let pSyms: &mut[tdefl_sym_freq];
    let mut num_used_syms: usize = 0;
    let pSym_count: &[u16] = &d.m_huff_count[table_num];
    while i < table_len { if pSym_count[i]!=0 { syms0[num_used_syms].m_key = pSym_count[i] as u16; syms0[num_used_syms].m_sym_index = i as u16; num_used_syms+=1}; i+=1 }

    pSyms = tdefl_radix_sort_syms(num_used_syms, &mut syms0, &mut syms1); tdefl_calculate_minimum_redundancy(pSyms, num_used_syms);

    i = 0;
    while i < num_used_syms { num_codes[pSyms[i].m_key as usize]+=1; i+=1}

    tdefl_huffman_enforce_max_code_size(&mut num_codes, num_used_syms, code_size_limit);

    for i in d.m_huff_code_sizes[table_num].iter_mut() {*i = 0u8};
    for i in d.m_huff_codes[table_num].iter_mut() {*i = 0u16};
    let mut j: usize;
    i = 1; j = num_used_syms;
    while i <= code_size_limit {
      let mut l = num_codes[i];
      while l > 0 { j-=1; d.m_huff_code_sizes[table_num][pSyms[j].m_sym_index as usize] = i as u8; l-=1 }
      i+=1
    }
  }

  next_code[1] = 0; let mut j = 0; i = 2; while i <= code_size_limit { j = (j + num_codes[i - 1]) << 1; next_code[i] = j; i+=1 };

  i = 0;
  while i < table_len
  {
    let mut rev_code: usize = 0; let mut code: usize; let code_size: usize = d.m_huff_code_sizes[table_num][i] as usize;
    if code_size == 0 {continue;}
    code = next_code[code_size]; next_code[code_size]+=1; let mut l = code_size; while l > 0 { rev_code = (rev_code << 1) | (code & 1); l-=1; code >>= 1};
    d.m_huff_codes[table_num][i] = rev_code as u16;
    i+=1
  }
}

const s_tdefl_packed_code_size_syms_swizzle: [u8; 19] = [ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ];

unsafe fn tdefl_start_dynamic_block(d: &mut tdefl_compressor)
{
  let mut num_lit_codes: usize = 286;
  let mut num_dist_codes: usize = 30;
  let mut num_bit_lengths: usize = 18;
  let mut i: usize;
  let total_code_sizes_to_pack: usize;
  let mut num_packed_code_sizes: usize = 0;
  let mut rle_z_count: c_uint = 0;
  let mut rle_repeat_count: c_uint = 0;
  let mut packed_code_sizes_index: usize = 0;
  let mut code_sizes_to_pack = [0u8; TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1];
  let mut packed_code_sizes = [0u8; TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1];
  let mut prev_code_size: u8 = 0xFF;

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: u64 = $b as u64; // WARNING what is size/proper type of bits?
      let len: usize = $l;
      assert!(bits <= ((1u64 << len) - 1u64));
      d.m_bit_buffer |= bits << d.m_bits_in; d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf = d.m_pOutput_buf.offset(1);
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  );

  macro_rules! TDEFL_RLE_PREV_CODE_SIZE( () =>
    ({
      if rle_repeat_count != 0 {
        if rle_repeat_count < 3 {
          d.m_huff_count[2][prev_code_size as usize] = d.m_huff_count[2][prev_code_size as usize] + (rle_repeat_count as u16);
          while {let temp = rle_repeat_count; rle_repeat_count -= 1; temp != 0} {
            packed_code_sizes[num_packed_code_sizes] = prev_code_size;
            num_packed_code_sizes += 1;
          }
        } else {
          d.m_huff_count[2][16] = (d.m_huff_count[2][16] + 1) as u16;
          packed_code_sizes[num_packed_code_sizes] = 16; num_packed_code_sizes += 1;
          packed_code_sizes[num_packed_code_sizes] = (rle_repeat_count - 3) as u8; num_packed_code_sizes += 1;
        } rle_repeat_count = 0;
      }
    });
  );

  macro_rules! TDEFL_RLE_ZERO_CODE_SIZE( () =>
    ({
      if rle_z_count != 0 {
        if rle_z_count < 3 {
          d.m_huff_count[2][0] = d.m_huff_count[2][0] + (rle_z_count as u16);
          while {let temp = rle_z_count; rle_z_count -= 1; temp != 0} {
            packed_code_sizes[num_packed_code_sizes] = 0; num_packed_code_sizes += 1;
          }
        } else if rle_z_count <= 10 {
          d.m_huff_count[2][17] = (d.m_huff_count[2][17] + 1) as u16;
          packed_code_sizes[num_packed_code_sizes] = 17; num_packed_code_sizes += 1;
          packed_code_sizes[num_packed_code_sizes] = (rle_z_count - 3) as u8; num_packed_code_sizes += 1;
        } else {
          d.m_huff_count[2][18] = (d.m_huff_count[2][18] + 1) as u16;
          packed_code_sizes[num_packed_code_sizes] = 18; num_packed_code_sizes += 1;
          packed_code_sizes[num_packed_code_sizes] = (rle_z_count - 11) as u8; num_packed_code_sizes += 1;
        } rle_z_count = 0;
      }
    });
  );

  d.m_huff_count[0][256] = 1;

  tdefl_optimize_huffman_table(d, 0, TDEFL_MAX_HUFF_SYMBOLS_0, 15, false);
  tdefl_optimize_huffman_table(d, 1, TDEFL_MAX_HUFF_SYMBOLS_1, 15, false);

  while num_lit_codes > 257 { if d.m_huff_code_sizes[0][num_lit_codes - 1]!=0 {break;}; num_lit_codes-=1; }
  while num_dist_codes > 1 { if d.m_huff_code_sizes[1][num_dist_codes - 1]!=0 {break;}; num_dist_codes-=1; }

  copy(&d.m_huff_code_sizes[0][0], code_sizes_to_pack.as_mut_ptr(), num_lit_codes);
  copy(&d.m_huff_code_sizes[1][0], code_sizes_to_pack.as_mut_ptr().offset(num_lit_codes as isize), num_dist_codes);
  total_code_sizes_to_pack = num_lit_codes + num_dist_codes;

  for i in d.m_huff_count[2][..TDEFL_MAX_HUFF_SYMBOLS_2].iter_mut() {*i = 0u16};
  i = 0;
  while i < total_code_sizes_to_pack
  {
    let code_size: u8 = code_sizes_to_pack[i];
    if code_size == 0 // !code_size
    {
      TDEFL_RLE_PREV_CODE_SIZE!();
      rle_z_count+=1;
      if rle_z_count == 138 { TDEFL_RLE_ZERO_CODE_SIZE!(); }
    }
    else
    {
      TDEFL_RLE_ZERO_CODE_SIZE!();
      if code_size != prev_code_size
      {
        TDEFL_RLE_PREV_CODE_SIZE!();
        d.m_huff_count[2][code_size as usize] = (d.m_huff_count[2][code_size as usize] + 1) as u16; packed_code_sizes[num_packed_code_sizes] = code_size; num_packed_code_sizes+=1;
      }
      else {
        rle_repeat_count+=1;
        if rle_repeat_count == 6
        {
          TDEFL_RLE_PREV_CODE_SIZE!();
        }
      }
    }
    prev_code_size = code_size;
    i+=1
  }
  if rle_repeat_count != 0 {
    TDEFL_RLE_PREV_CODE_SIZE!();
  } else {
    TDEFL_RLE_ZERO_CODE_SIZE!();
  }

  tdefl_optimize_huffman_table(d, 2, TDEFL_MAX_HUFF_SYMBOLS_2, 7, false);

  TDEFL_PUT_BITS!(2, 2);

  TDEFL_PUT_BITS!(num_lit_codes - 257, 5);
  TDEFL_PUT_BITS!(num_dist_codes - 1, 5);

  while num_bit_lengths >= 0 { if d.m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[num_bit_lengths] as usize] != 0 {break;}; num_bit_lengths-=1 }
  num_bit_lengths = max(4, (num_bit_lengths + 1)); TDEFL_PUT_BITS!(num_bit_lengths - 4, 4);
  i = 0;
  while i < num_bit_lengths { TDEFL_PUT_BITS!(d.m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[i] as usize] as usize, 3); i+=1 }

  while packed_code_sizes_index < num_packed_code_sizes
  {
    let code: usize = packed_code_sizes[packed_code_sizes_index] as usize; packed_code_sizes_index+=1; assert!(code < TDEFL_MAX_HUFF_SYMBOLS_2);
    TDEFL_PUT_BITS!(d.m_huff_codes[2][code] as usize, d.m_huff_code_sizes[2][code] as usize);
    if code >= 16 {TDEFL_PUT_BITS!(packed_code_sizes[packed_code_sizes_index] as usize, [02, 03, 07][code - 16]); packed_code_sizes_index+=1;}
  }
}

unsafe fn tdefl_start_static_block(d: &mut tdefl_compressor)
{
  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: u64 = $b as u64; // WARNING what is size/proper type of bits?
      let len: usize = $l;
      assert!(bits <= ((1u64 << len) - 1u64));
      d.m_bit_buffer |= bits << d.m_bits_in; d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf = d.m_pOutput_buf.offset(1);
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  );

  for i in d.m_huff_code_sizes[0][  0..144].iter_mut() { *i = 8 }
  for i in d.m_huff_code_sizes[0][144..256].iter_mut() { *i = 9 }
  for i in d.m_huff_code_sizes[0][256..280].iter_mut() { *i = 7 }
  for i in d.m_huff_code_sizes[0][280..288].iter_mut() { *i = 8 }

  for i in d.m_huff_code_sizes[1][  0.. 32].iter_mut() { *i = 5 }

  tdefl_optimize_huffman_table(d, 0, 288, 15, true);
  tdefl_optimize_huffman_table(d, 1, 32, 15, true);

  TDEFL_PUT_BITS!(1, 2);
}

const MZ_BITMASKS: [u16; 17] = [ 0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF ];

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
unsafe fn tdefl_compress_lz_codes(d: &mut tdefl_compressor) -> bool
{
  let mut flags: usize;
  let mut pLZ_codes: *const u8 = d.m_lz_code_buf.as_ptr();
  let mut pOutput_buf: *mut u8 = d.m_pOutput_buf;
  let pLZ_code_buf_end: *const u8 = d.m_pLZ_code_buf as *const u8;
  let mut bit_buffer: u64 = d.m_bit_buffer;
  let mut bits_in: usize = d.m_bits_in;

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: u64 = $b; // WARNING what is size/proper type of bits?
      let len: usize = $l;
      assert!(bits <= ((1u64 << len) - 1u64));
      d.m_bit_buffer |= bits << d.m_bits_in; d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf = d.m_pOutput_buf.offset(1);
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  );

  macro_rules! TDEFL_PUT_BITS_FAST( ($b:expr, $l:expr) =>
    ({
      bit_buffer |= ($b as u64) << bits_in;
      bits_in += $l;
    });
  );

  flags = 1;
  while (pLZ_codes as usize) < (pLZ_code_buf_end as usize)
  {
    if flags == 1 {
      flags = *pLZ_codes as usize | 0x100;
      pLZ_codes = pLZ_codes.offset(1);
    }

    if flags & 1 != 0
    {
      let match_len: usize = *(pLZ_codes.offset(0)) as usize;
      let match_dist: usize = *(pLZ_codes.offset(1) as *const u16) as usize; pLZ_codes = pLZ_codes.offset(3);

      assert!(d.m_huff_code_sizes[0][S_TDEFL_LEN_SYM[match_len] as usize] != 0);
      TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][S_TDEFL_LEN_SYM[match_len] as usize], d.m_huff_code_sizes[0][S_TDEFL_LEN_SYM[match_len] as usize] as usize);
      TDEFL_PUT_BITS_FAST!(match_len & MZ_BITMASKS[S_TDEFL_LEN_EXTRA[match_len] as usize] as usize, S_TDEFL_LEN_EXTRA[match_len] as usize);

      // This sequence coaxes MSVC into using cmov's vs. jmp's.
      let s0: u8 = S_TDEFL_SMALL_DIST_SYM[match_dist & 511];
      let s1: u8 = S_TDEFL_SMALL_DIST_EXTRA[match_dist & 511];
      let n0: u8 = S_TDEFL_LARGE_DIST_SYM[match_dist >> 8];
      let n1: u8 = S_TDEFL_LARGE_DIST_EXTRA[match_dist >> 8];
      let sym: usize = if match_dist < 512 {s0} else {s1} as usize;
      let num_extra_bits: usize = if match_dist < 512 {n0} else {n1} as usize;

      assert!(d.m_huff_code_sizes[1][sym] != 0);
      TDEFL_PUT_BITS_FAST!(d.m_huff_codes[1][sym], d.m_huff_code_sizes[1][sym] as usize);
      TDEFL_PUT_BITS_FAST!(match_dist & MZ_BITMASKS[num_extra_bits] as usize, num_extra_bits);
    }
    else
    {
      let mut lit: usize = *pLZ_codes as usize;
      pLZ_codes = pLZ_codes.offset(1);
      assert!(d.m_huff_code_sizes[0][lit] != 0);
      TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit] as usize);

      if ((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end)
      {
        flags >>= 1;
        lit = *pLZ_codes as usize;
        pLZ_codes = pLZ_codes.offset(1);
        assert!(d.m_huff_code_sizes[0][lit] != 0);
        TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit] as usize);

        if ((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end)
        {
          flags >>= 1;
          lit = *pLZ_codes as usize;
          pLZ_codes = pLZ_codes.offset(1);
          assert!(d.m_huff_code_sizes[0][lit] != 0);
          TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit] as usize);
        }
      }
    }

    if pOutput_buf >= d.m_pOutput_buf_end {
      return false;
    }

    *(pOutput_buf as *mut u64) = bit_buffer;
    pOutput_buf = pOutput_buf.offset(bits_in as isize >> 3);
    bit_buffer >>= bits_in & !7;
    bits_in &= 7;

    flags >>= 1;
  }

  d.m_pOutput_buf = pOutput_buf;
  d.m_bits_in = 0;
  d.m_bit_buffer = 0;

  while bits_in != 0
  {
    let n: usize = min(bits_in, 16);
    TDEFL_PUT_BITS!((bit_buffer) & MZ_BITMASKS[n] as u64, n);
    bit_buffer >>= n;
    bits_in -= n;
  }

  TDEFL_PUT_BITS!(d.m_huff_codes[0][256] as u64, d.m_huff_code_sizes[0][256] as usize);

  return d.m_pOutput_buf < d.m_pOutput_buf_end;
}
#[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
unsafe fn tdefl_compress_lz_codes(d: &mut tdefl_compressor) -> bool
{
  let flags: c_uint = 1;
  let pLZ_codes: *const u8 = d.m_lz_code_buf;

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: c_uint = $b;
      let len: c_uint = $l;
      assert!(bits <= ((1u << len) - 1u));
      d.m_bit_buffer |= (bits << d.m_bits_in); d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf += 1;
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  );

  while pLZ_codes < d.m_pLZ_code_buf
  {
    if (flags == 1){
      flags = *pLZ_codes | 0x100;
      pLZ_codes += 1;
    }
    if (flags & 1)
    {
      let sym: c_uint; let num_extra_bits: c_uint;
      let match_len: c_uint = pLZ_codes[0]; let match_dist: c_uint = (pLZ_codes[1] | (pLZ_codes[2] << 8)); pLZ_codes += 3;

      assert!(d.m_huff_code_sizes[0][S_TDEFL_LEN_SYM[match_len]]);
      TDEFL_PUT_BITS!(d.m_huff_codes[0][S_TDEFL_LEN_SYM[match_len]], d.m_huff_code_sizes[0][S_TDEFL_LEN_SYM[match_len]]);
      TDEFL_PUT_BITS!(match_len & MZ_BITMASKS[S_TDEFL_LEN_EXTRA[match_len]], S_TDEFL_LEN_EXTRA[match_len]);

      if (match_dist < 512)
      {
        sym = S_TDEFL_SMALL_DIST_SYM[match_dist]; num_extra_bits = S_TDEFL_SMALL_DIST_EXTRA[match_dist];
      }
      else
      {
        sym = S_TDEFL_LARGE_DIST_SYM[match_dist >> 8]; num_extra_bits = S_TDEFL_LARGE_DIST_EXTRA[match_dist >> 8];
      }
      assert!(d.m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS!(d.m_huff_codes[1][sym], d.m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS!(match_dist & MZ_BITMASKS[num_extra_bits], num_extra_bits);
    }
    else
    {
      let lit: c_uint = *pLZ_codes; pLZ_codes+=1;
      assert!(d.m_huff_code_sizes[0][lit]);
      TDEFL_PUT_BITS!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit]);
    }
    flags >>= 1;
  }

  TDEFL_PUT_BITS!(d.m_huff_codes[0][256], d.m_huff_code_sizes[0][256]);

  return (d.m_pOutput_buf < d.m_pOutput_buf_end);
}
// #endif // MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN && MINIZ_HAS_64BIT_REGISTERS

unsafe fn tdefl_compress_block(d: &mut tdefl_compressor, static_block: bool) -> bool
{
  if static_block {
    tdefl_start_static_block(d);
  } else {
    tdefl_start_dynamic_block(d);
  } return tdefl_compress_lz_codes(d);
}

unsafe fn tdefl_flush_block(d: &mut tdefl_compressor, flush: tdefl_flush) -> tdefl_status
{
  let saved_bit_buf: u64; let saved_bits_in: usize;
  let pSaved_output_buf: *mut u8;
  let mut comp_block_succeeded: bool = false;
  let use_raw_block: bool = d.m_flags.contains(TDEFL_FORCE_ALL_RAW_BLOCKS) && (d.m_lookahead_pos - d.m_lz_code_buf_dict_pos <= d.m_dict_size);
  let pOutput_buf_start: *mut u8 = if (d.m_pPut_buf_func.is_none()) && ((*d.m_pOut_buf_size - d.m_out_buf_ofs) >= TDEFL_OUT_BUF_SIZE) {
    (d.m_pOut_buf).offset(d.m_out_buf_ofs as isize)
  } else {
    d.m_output_buf.as_mut_ptr()
  };

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: u64 = $b;
      let len: usize = $l;
      assert!(bits <= ((1u64 << len) - 1u64));
      d.m_bit_buffer |= bits << d.m_bits_in; d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf = d.m_pOutput_buf.offset(1);
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  );

  d.m_pOutput_buf = pOutput_buf_start;
  d.m_pOutput_buf_end = d.m_pOutput_buf.offset(TDEFL_OUT_BUF_SIZE as isize- 16);

  assert!(d.m_output_flush_remaining == 0);
  d.m_output_flush_ofs = 0;
  d.m_output_flush_remaining = 0;

  *d.m_pLZ_flags = (*d.m_pLZ_flags >> d.m_num_flags_left as usize) as u8;
  d.m_pLZ_code_buf.offset(- ((d.m_num_flags_left == 8) as isize));

  if d.m_flags.contains(TDEFL_WRITE_ZLIB_HEADER) && d.m_block_index == 0
  {
    TDEFL_PUT_BITS!(0x78, 8); TDEFL_PUT_BITS!(0x01, 8);
  }

  TDEFL_PUT_BITS!((flush == TDEFL_FINISH) as u64, 1);

  pSaved_output_buf = d.m_pOutput_buf; saved_bit_buf = d.m_bit_buffer; saved_bits_in = d.m_bits_in;

  if !use_raw_block {
    let _borrow_temp = d.m_flags.contains(TDEFL_FORCE_ALL_STATIC_BLOCKS) || (d.m_total_lz_bytes < 48);
    comp_block_succeeded = tdefl_compress_block(d, _borrow_temp);
  }

  // If the block gets expanded, forget the current contents of the output buffer and send a raw block instead.
  if ((use_raw_block) || ((d.m_total_lz_bytes != 0) && ((d.m_pOutput_buf as usize - pSaved_output_buf as usize + 1) >= d.m_total_lz_bytes))) &&
       ((d.m_lookahead_pos - d.m_lz_code_buf_dict_pos) <= d.m_dict_size)
  {
    let mut i: usize; d.m_pOutput_buf = pSaved_output_buf; d.m_bit_buffer = saved_bit_buf; d.m_bits_in = saved_bits_in;
    TDEFL_PUT_BITS!(0, 2);
    if d.m_bits_in != 0 { TDEFL_PUT_BITS!(0, 8 - d.m_bits_in); }
    i = 2;
    while i > 0
    {
      TDEFL_PUT_BITS!(d.m_total_lz_bytes as u64 & 0xFFFF, 16);
      i -= 1; d.m_total_lz_bytes ^= 0xFFFF;
    }
    i = 0;
    while i < d.m_total_lz_bytes
    {
      TDEFL_PUT_BITS!(d.m_dict[(d.m_lz_code_buf_dict_pos + i) & TDEFL_LZ_DICT_SIZE_MASK] as u64, 8);
      i += 1;
    }
  }
  // Check for the extremely unlikely (if not impossible) case of the compressed block not fitting into the output buffer when using dynamic codes.
  else if !comp_block_succeeded
  {
    d.m_pOutput_buf = pSaved_output_buf; d.m_bit_buffer = saved_bit_buf; d.m_bits_in = saved_bits_in;
    tdefl_compress_block(d, true);
  }

  if flush != TDEFL_NO_FLUSH
  {
    if flush == TDEFL_FINISH
    {
      if d.m_bits_in != 0 { TDEFL_PUT_BITS!(0, 8 - d.m_bits_in); }
      if d.m_flags.contains(TDEFL_WRITE_ZLIB_HEADER) {
        let mut i: c_uint = 0; let mut a: c_uint = d.m_adler32;
        while i < 4 { TDEFL_PUT_BITS!((a as u64 >> 24) & 0xFF, 8); a <<= 8; i+=1; }
      }
    }
    else
    {
      let mut i: c_uint = 2; let mut z: u64 = 0; TDEFL_PUT_BITS!(0, 3);
      if d.m_bits_in != 0 { TDEFL_PUT_BITS!(0, 8 - d.m_bits_in); }
      while i > 0 { TDEFL_PUT_BITS!(z & 0xFFFF, 16); i -= 1; z ^= 0xFFFF; }
    }
  }

  assert!(d.m_pOutput_buf < d.m_pOutput_buf_end);

  for i in d.m_huff_count[0][..TDEFL_MAX_HUFF_SYMBOLS_0].iter_mut() { *i = 0; } // TODO check bounds
  for i in d.m_huff_count[1][..TDEFL_MAX_HUFF_SYMBOLS_1].iter_mut() { *i = 0; }

  d.m_pLZ_code_buf = d.m_lz_code_buf.as_mut_ptr().offset(1);
  d.m_pLZ_flags = d.m_lz_code_buf.as_mut_ptr();
  d.m_num_flags_left = 8;
  d.m_lz_code_buf_dict_pos += d.m_total_lz_bytes;
  d.m_total_lz_bytes = 0;
  d.m_block_index+=1;

  let mut n: usize = d.m_pOutput_buf as usize - pOutput_buf_start as usize; // WARNING ptrdiff_t
  if n != 0
  {
    if let Some(ref mut put_buf_func) = d.m_pPut_buf_func
    {
      *d.m_pIn_buf_size = d.m_pSrc as usize - d.m_pIn_buf as usize; // WARNING ptrdiff_t
      if ! put_buf_func.call_mut((d.m_output_buf.as_ptr(), n)) {
        d.m_prev_return_status = TDEFL_STATUS_PUT_BUF_FAILED;
        return d.m_prev_return_status;
      }
    }
    else if pOutput_buf_start as *const u8 == d.m_output_buf.as_ptr()
    {
      let bytes_to_copy: usize = min(n as usize, (*d.m_pOut_buf_size - d.m_out_buf_ofs));
      copy(d.m_output_buf.as_ptr(), d.m_pOut_buf.offset(d.m_out_buf_ofs as isize), bytes_to_copy);
      d.m_out_buf_ofs += bytes_to_copy;

      n -= bytes_to_copy;
      if n != 0
      {
        d.m_output_flush_ofs = bytes_to_copy;
        d.m_output_flush_remaining = n;
      }
    }
    else
    {
      d.m_out_buf_ofs += n;
    }
  }

  return match d.m_output_flush_remaining {
    0 => TDEFL_STATUS_OKAY,
    _ => TDEFL_STATUS_DONE
  };
}

// #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES
macro_rules! TDEFL_READ_UNALIGNED_WORD(($p:expr) => (*($p as *const u16)); );
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tdefl_find_match(d: &mut tdefl_compressor, lookahead_pos: usize, max_dist: usize, max_match_len: usize, pMatch_dist: &mut usize, pMatch_len: &mut usize)
{
  let mut dist: usize;
  let pos: usize = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
  let mut match_len: usize = *pMatch_len;
  let mut probe_pos: usize = pos;
  let mut next_probe_pos: usize;
  let mut probe_len: usize;
  let mut num_probes_left: usize = d.m_max_probes[(match_len >= 32) as usize];
  let s: *const u16 = d.m_dict.as_ptr().offset(pos as isize) as *const u16;
  let mut p: *const u16;
  let mut q: *const u16;
  let mut c01: u16 = TDEFL_READ_UNALIGNED_WORD!(d.m_dict.as_ptr().offset((pos + match_len - 1) as isize));
  let s01: u16 = TDEFL_READ_UNALIGNED_WORD!(s);
  assert!(max_match_len <= TDEFL_MAX_MATCH_LEN); if max_match_len <= match_len {return;}
  loop {
    loop {
      num_probes_left -= 1;
      if num_probes_left == 0 {return;}
      macro_rules! TDEFL_PROBE( () => ({
        next_probe_pos = d.m_next[probe_pos] as usize;
        dist = lookahead_pos - next_probe_pos;
        if (next_probe_pos == 0) || (dist > max_dist) {return;}
        probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;
        if TDEFL_READ_UNALIGNED_WORD!(d.m_dict.as_ptr().offset((probe_pos + match_len - 1) as isize)) == c01 {break;}
      }););
      TDEFL_PROBE!(); TDEFL_PROBE!(); TDEFL_PROBE!();
    }
    if dist==0 {break;};
    q = d.m_dict.as_ptr().offset(probe_pos as isize) as *const u16;
    if TDEFL_READ_UNALIGNED_WORD!(q) != s01 {continue;};
    p = s; probe_len = 32;
    loop {
      if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
        if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
          if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
            if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
              if {probe_len -= 1; probe_len > 0} {
                continue;
              }
            }
          }
        }
      }
      break;
    }
    if probe_len == 0
    {
      *pMatch_dist = dist; *pMatch_len = min(max_match_len, TDEFL_MAX_MATCH_LEN); break;
    }
    else if {probe_len = (((p as usize - s as usize)) * 2) + (*(p as *const u8) == *(q as *const u8)) as usize; probe_len > match_len}
    {
      *pMatch_dist = dist; match_len = min(max_match_len, probe_len); *pMatch_len = match_len;
      if match_len == max_match_len {break;}
      c01 = TDEFL_READ_UNALIGNED_WORD!(d.m_dict.as_ptr().offset((pos + match_len - 1) as isize));
    }
  }
}
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn tdefl_find_match(d: &mut tdefl_compressor, lookahead_pos: usize, max_dist: usize, max_match_len: usize, pMatch_dist: &mut usize, pMatch_len: &mut usize)
{
  let dist: c_uint;
  let pos: c_uint = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
  let match_len: c_uint = *pMatch_len;
  let probe_pos: c_uint = pos;
  let next_probe_pos: c_uint;
  let probe_len: c_uint;
  let num_probes_left: c_uint = d.m_max_probes[match_len >= 32];
  let s: *const u8 = d.m_dict + pos;
  let p: *const u8;
  let q: *const u8;
  let c0: u8 = d.m_dict[pos + match_len];
  let c1: u8 = d.m_dict[pos + match_len - 1];
  assert!(max_match_len <= TDEFL_MAX_MATCH_LEN); if max_match_len <= match_len {return;}
  loop
  {
    loop
    {
      num_probes_left -= 1;
      if num_probes_left == 0 {return;}
      macro_rules! TDEFL_PROBE( () => (
        next_probe_pos = d->m_next[probe_pos];
        if ((!next_probe_pos) || ((dist = (mz_uint16)(lookahead_pos - next_probe_pos)) > max_dist)) return;
        probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;
        if ((d->m_dict[probe_pos + match_len] == c0) && (d->m_dict[probe_pos + match_len - 1] == c1)) break;
      ););
      TDEFL_PROBE; TDEFL_PROBE; TDEFL_PROBE;
    }
    if !dist {break;} p = s; q = d.m_dict + probe_pos;
    probe_len = 0;
    while probe_len < max_match_len {
      if {let temp = *p != *q; p += 1; q += 1; temp} {break;};
      probe_len += 1;
    }
    if (probe_len > match_len)
    {
      *pMatch_dist = dist; if (*pMatch_len = match_len = probe_len) == max_match_len {return;}
      c0 = d.m_dict[pos + match_len]; c1 = d.m_dict[pos + match_len - 1];
    }
  }
}
// #endif // #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES

// #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
unsafe fn tdefl_compress_fast(d: &mut tdefl_compressor) -> bool
{
  // Faster, minimally featured LZRW1-style match+parse loop with better register utilization. Intended for applications where raw throughput is valued more highly than ratio.
  let mut lookahead_pos: usize = d.m_lookahead_pos;
  let mut lookahead_size: usize = d.m_lookahead_size;
  let mut dict_size: usize = d.m_dict_size;
  let mut total_lz_bytes: usize = d.m_total_lz_bytes;
  let mut num_flags_left: usize = d.m_num_flags_left;
  let mut pLZ_code_buf: *mut u8 = d.m_pLZ_code_buf; let mut pLZ_flags: *mut u8 = d.m_pLZ_flags;
  let mut cur_pos: usize = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;

  while d.m_src_buf_left > 0 || ((d.m_flush != TDEFL_NO_FLUSH) && (lookahead_size > 0))
  {
    let TDEFL_COMP_FAST_LOOKAHEAD_SIZE: usize = 4096;
    let mut dst_pos: usize = (lookahead_pos + lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
    let mut num_bytes_to_process: usize = min(d.m_src_buf_left, TDEFL_COMP_FAST_LOOKAHEAD_SIZE - lookahead_size);
    d.m_src_buf_left -= num_bytes_to_process;
    lookahead_size += num_bytes_to_process;

    while num_bytes_to_process > 0
    {
      let n: usize = min(TDEFL_LZ_DICT_SIZE - dst_pos, num_bytes_to_process);
      copy(d.m_pSrc, d.m_dict.as_mut_ptr().offset(dst_pos as isize), n);
      if dst_pos < (TDEFL_MAX_MATCH_LEN - 1) {
        copy(d.m_pSrc, d.m_dict.as_mut_ptr().offset((TDEFL_LZ_DICT_SIZE + dst_pos) as isize), min(n, (TDEFL_MAX_MATCH_LEN - 1) - dst_pos));
      }
      d.m_pSrc = d.m_pSrc.offset(n as isize);
      dst_pos = (dst_pos + n) & TDEFL_LZ_DICT_SIZE_MASK;
      num_bytes_to_process -= n;
    }

    dict_size = min(TDEFL_LZ_DICT_SIZE - lookahead_size, dict_size);
    if (d.m_flush == TDEFL_NO_FLUSH) && (lookahead_size < TDEFL_COMP_FAST_LOOKAHEAD_SIZE) {break;}

    while lookahead_size >= 4
    {
      let mut cur_match_dist: usize;
      let mut cur_match_len: usize = 1;
      let pCur_dict: *const u8 = d.m_dict.as_ptr().offset(cur_pos as isize);
      let first_trigram: usize = (*(pCur_dict as *const u32)) as usize & 0xFFFFFF;
      let hash: usize = (first_trigram ^ (first_trigram >> (24 - (TDEFL_LZ_HASH_BITS - 8)))) & TDEFL_LEVEL1_HASH_SIZE_MASK;
      let mut probe_pos: usize = d.m_hash[hash] as usize;
      d.m_hash[hash] = lookahead_pos as u16;

      if {cur_match_dist = lookahead_pos - probe_pos; cur_match_dist <= dict_size} &&
         {probe_pos &= TDEFL_LZ_DICT_SIZE_MASK; *(d.m_dict.as_ptr().offset(probe_pos as isize) as *const u32) as usize & 0xFFFFFF == first_trigram}
      {
        let mut p: *const u16 = pCur_dict as *const u16;
        let mut q: *const u16 = d.m_dict.as_ptr().offset(probe_pos as isize) as *const u16;
        let mut probe_len: u32 = 32;
        loop {
          if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
            if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
              if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
                if {p = p.offset(1); q = q.offset(1); (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
                  if {probe_len -= 1; probe_len > 0} {
                    continue;
                  }
                }
              }
            }
          }
          break;
        }
        cur_match_len = ((p as usize - (pCur_dict as *const u16) as usize) * 2) + (*(p as *const u8) == *(q as *const u8)) as usize;
        if probe_len == 0 {
          cur_match_len = if cur_match_dist > 0 {TDEFL_MAX_MATCH_LEN} else {0};
        }

        if (cur_match_len < TDEFL_MIN_MATCH_LEN) || ((cur_match_len == TDEFL_MIN_MATCH_LEN) && (cur_match_dist >= 8usize*1024))
        {
          cur_match_len = 1;
          *pLZ_code_buf = first_trigram as u8; pLZ_code_buf = pLZ_code_buf.offset(1);
          *pLZ_flags = (*pLZ_flags >> 1) as u8;
          d.m_huff_count[0][(first_trigram as u8) as usize]+=1;
        }
        else
        {
          let s0: usize; let s1: usize;
          cur_match_len = min(cur_match_len, lookahead_size);

          assert!((cur_match_len >= TDEFL_MIN_MATCH_LEN) && (cur_match_dist >= 1) && (cur_match_dist <= TDEFL_LZ_DICT_SIZE));

          cur_match_dist-=1;

          *pLZ_code_buf = (cur_match_len - TDEFL_MIN_MATCH_LEN) as u8;
          *(pLZ_code_buf.offset(1) as *mut u16) = cur_match_dist as u16;
          pLZ_code_buf = pLZ_code_buf.offset(3);
          *pLZ_flags = ((*pLZ_flags >> 1) | 0x80) as u8;

          s0 = S_TDEFL_SMALL_DIST_SYM[cur_match_dist & 511] as usize;
          s1 = S_TDEFL_LARGE_DIST_SYM[cur_match_dist >> 8] as usize;
          d.m_huff_count[1][if cur_match_dist < 512 {s0} else {s1}]+=1;

          d.m_huff_count[0][S_TDEFL_LEN_SYM[cur_match_len - TDEFL_MIN_MATCH_LEN] as usize]+=1;
        }
      }
      else
      {
        *pLZ_code_buf = first_trigram as u8; pLZ_code_buf = pLZ_code_buf.offset(1);
        *pLZ_flags = (*pLZ_flags >> 1) as u8;
        d.m_huff_count[0][(first_trigram as u8) as usize]+=1;
      }

      num_flags_left -= 1;
      if num_flags_left == 0 { num_flags_left = 8; pLZ_flags = pLZ_code_buf; pLZ_code_buf = pLZ_code_buf.offset(1); }

      total_lz_bytes += cur_match_len;
      lookahead_pos += cur_match_len;
      dict_size = min(dict_size + cur_match_len, TDEFL_LZ_DICT_SIZE);
      cur_pos = (cur_pos + cur_match_len) & TDEFL_LZ_DICT_SIZE_MASK;
      assert!(lookahead_size >= cur_match_len);
      lookahead_size -= cur_match_len;

      if pLZ_code_buf as *const u8 > d.m_lz_code_buf.as_ptr().offset(TDEFL_LZ_CODE_BUF_SIZE as isize - 8)
      {
        let n: i8;
        d.m_lookahead_pos = lookahead_pos; d.m_lookahead_size = lookahead_size; d.m_dict_size = dict_size;
        d.m_total_lz_bytes = total_lz_bytes; d.m_pLZ_code_buf = pLZ_code_buf; d.m_pLZ_flags = pLZ_flags; d.m_num_flags_left = num_flags_left;
        if {n = tdefl_flush_block(d, TDEFL_NO_FLUSH) as i8; n != 0} {
          return if n < 0 {false} else {true};
        }
        total_lz_bytes = d.m_total_lz_bytes; pLZ_code_buf = d.m_pLZ_code_buf; pLZ_flags = d.m_pLZ_flags; num_flags_left = d.m_num_flags_left;
      }
    }

    while lookahead_size > 0
    {
      let lit: u8 = d.m_dict[cur_pos];

      total_lz_bytes+=1;
      *pLZ_code_buf = lit; pLZ_code_buf = pLZ_code_buf.offset(1);
      *pLZ_flags = (*pLZ_flags >> 1) as u8;
      num_flags_left -= 1;
      if num_flags_left == 0 { num_flags_left = 8; pLZ_flags = pLZ_code_buf; pLZ_code_buf = pLZ_code_buf.offset(1); }

      d.m_huff_count[0][lit as usize]+=1;

      lookahead_pos+=1;
      dict_size = min(dict_size + 1, TDEFL_LZ_DICT_SIZE);
      cur_pos = (cur_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK;
      lookahead_size-=1;

      if pLZ_code_buf as *const u8 > d.m_lz_code_buf.as_ptr().offset(TDEFL_LZ_CODE_BUF_SIZE as isize - 8)
      {
        let n: i8;
        d.m_lookahead_pos = lookahead_pos; d.m_lookahead_size = lookahead_size; d.m_dict_size = dict_size;
        d.m_total_lz_bytes = total_lz_bytes; d.m_pLZ_code_buf = pLZ_code_buf; d.m_pLZ_flags = pLZ_flags; d.m_num_flags_left = num_flags_left;
        if {n = tdefl_flush_block(d, TDEFL_NO_FLUSH) as i8; n != 0} {
          return if n < 0 {false} else {true};
        }
        total_lz_bytes = d.m_total_lz_bytes; pLZ_code_buf = d.m_pLZ_code_buf; pLZ_flags = d.m_pLZ_flags; num_flags_left = d.m_num_flags_left;
      }
    }
  }

  d.m_lookahead_pos = lookahead_pos; d.m_lookahead_size = lookahead_size; d.m_dict_size = dict_size;
  d.m_total_lz_bytes = total_lz_bytes; d.m_pLZ_code_buf = pLZ_code_buf; d.m_pLZ_flags = pLZ_flags; d.m_num_flags_left = num_flags_left;
  return true;
}
// #endif // MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN

#[inline(always)]
unsafe fn tdefl_record_literal(d: &mut tdefl_compressor, lit: u8)
{
  d.m_total_lz_bytes+=1;
  *d.m_pLZ_code_buf = lit; d.m_pLZ_code_buf = d.m_pLZ_code_buf.offset(1);
  *d.m_pLZ_flags = (*d.m_pLZ_flags >> 1) as u8; d.m_num_flags_left -= 1;
  if d.m_num_flags_left == 0 { d.m_num_flags_left = 8; d.m_pLZ_flags = d.m_pLZ_code_buf; d.m_pLZ_code_buf = d.m_pLZ_code_buf.offset(1); }
  d.m_huff_count[0][lit as usize] += 1;
}

#[inline(always)]
unsafe fn tdefl_record_match(d: &mut tdefl_compressor, match_len: usize, mut match_dist: usize)
{
  let s0: u8;
  let s1: u8;

  assert!((match_len >= TDEFL_MIN_MATCH_LEN) && (match_dist >= 1) && (match_dist <= TDEFL_LZ_DICT_SIZE));

  d.m_total_lz_bytes += match_len;

  *d.m_pLZ_code_buf.offset(0) = (match_len - TDEFL_MIN_MATCH_LEN) as u8;
  match_dist -= 1;
  *d.m_pLZ_code_buf.offset(1) = (match_dist & 0xFF) as u8;
  *d.m_pLZ_code_buf.offset(2) = (match_dist >> 8) as u8;
  d.m_pLZ_code_buf = d.m_pLZ_code_buf.offset(3);

  *d.m_pLZ_flags = ((*d.m_pLZ_flags >> 1) | 0x80) as u8; d.m_num_flags_left -= 1;
  if d.m_num_flags_left == 0 { d.m_num_flags_left = 8; d.m_pLZ_flags = d.m_pLZ_code_buf; d.m_pLZ_code_buf = d.m_pLZ_code_buf.offset(1); }

  s0 = S_TDEFL_SMALL_DIST_SYM[match_dist & 511];
  s1 = S_TDEFL_LARGE_DIST_SYM[(match_dist >> 8) & 127];
  d.m_huff_count[1][if match_dist < 512 {s0} else {s1} as usize] += 1;

  if match_len >= TDEFL_MIN_MATCH_LEN {d.m_huff_count[0][S_TDEFL_LEN_SYM[match_len - TDEFL_MIN_MATCH_LEN] as usize] += 1;}
}

unsafe fn tdefl_compress_normal(d: &mut tdefl_compressor) -> bool
{
  let mut pSrc: *const u8 = d.m_pSrc;
  let mut src_buf_left: usize = d.m_src_buf_left;
  let flush: tdefl_flush = d.m_flush;

  while (src_buf_left > 0) || ((flush != TDEFL_NO_FLUSH) && (d.m_lookahead_size > 0))
  {
    // Update dictionary and hash chains. Keeps the lookahead size equal to TDEFL_MAX_MATCH_LEN.
    if (d.m_lookahead_size + d.m_dict_size) >= (TDEFL_MIN_MATCH_LEN - 1)
    {
      let mut dst_pos: usize = (d.m_lookahead_pos + d.m_lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
      let mut ins_pos: usize = d.m_lookahead_pos + d.m_lookahead_size - 2;
      let mut hash: usize = ((d.m_dict[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] as usize) << TDEFL_LZ_HASH_SHIFT) ^ d.m_dict[(ins_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK] as usize;
      let num_bytes_to_process: usize = min(src_buf_left, TDEFL_MAX_MATCH_LEN - d.m_lookahead_size);
      let pSrc_end: *const u8 = pSrc.offset(num_bytes_to_process as isize);
      src_buf_left -= num_bytes_to_process;
      d.m_lookahead_size += num_bytes_to_process;
      while pSrc != pSrc_end
      {
        let c: u8 = *pSrc; pSrc = pSrc.offset(1); d.m_dict[dst_pos] = c;
        if dst_pos < (TDEFL_MAX_MATCH_LEN - 1) {d.m_dict[TDEFL_LZ_DICT_SIZE + dst_pos] = c;}
        hash = ((hash << TDEFL_LZ_HASH_SHIFT) ^ c as usize) & (TDEFL_LZ_HASH_SIZE - 1);
        d.m_next[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] = d.m_hash[hash]; d.m_hash[hash] = ins_pos as u16;
        dst_pos = (dst_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK; ins_pos += 1;
      }
    }
    else
    {
      while (src_buf_left > 0) && (d.m_lookahead_size < TDEFL_MAX_MATCH_LEN)
      {
        let c: u8 = *pSrc; pSrc = pSrc.offset(1);
        let dst_pos: usize = (d.m_lookahead_pos + d.m_lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
        src_buf_left -= 1;
        d.m_dict[dst_pos] = c;
        if dst_pos < (TDEFL_MAX_MATCH_LEN - 1) {
          d.m_dict[TDEFL_LZ_DICT_SIZE + dst_pos] = c;
        }
        d.m_lookahead_size += 1;
        if (d.m_lookahead_size + d.m_dict_size) >= TDEFL_MIN_MATCH_LEN
        {
          let ins_pos: usize = d.m_lookahead_pos + (d.m_lookahead_size - 1) - 2;
          let hash: usize = (((d.m_dict[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] as usize) << (TDEFL_LZ_HASH_SHIFT * 2)) ^ ((d.m_dict[(ins_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK] as usize) << TDEFL_LZ_HASH_SHIFT) ^ c as usize) & (TDEFL_LZ_HASH_SIZE - 1);
          d.m_next[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] = d.m_hash[hash]; d.m_hash[hash] = ins_pos as u16;
        }
      }
    }
    d.m_dict_size = min(TDEFL_LZ_DICT_SIZE - d.m_lookahead_size, d.m_dict_size);
    if (flush == TDEFL_NO_FLUSH) && (d.m_lookahead_size < TDEFL_MAX_MATCH_LEN) {
      break;
    }

    let mut len_to_move: usize;
    let mut cur_match_dist: usize;
    let mut cur_match_len: usize;
    let cur_pos: usize;
    // Simple lazy/greedy parsing state machine.
    len_to_move = 1; cur_match_dist = 0; cur_match_len = if d.m_saved_match_len>0 {d.m_saved_match_len} else {TDEFL_MIN_MATCH_LEN - 1}; cur_pos = d.m_lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
    if d.m_flags.contains(TDEFL_RLE_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS)
    {
      if (d.m_dict_size > 0) && (!d.m_flags.contains(TDEFL_FORCE_ALL_RAW_BLOCKS))
      {
        let c: u8 = d.m_dict[(cur_pos - 1) & TDEFL_LZ_DICT_SIZE_MASK];
        cur_match_len = 0; while cur_match_len < d.m_lookahead_size { if d.m_dict[cur_pos + cur_match_len] != c {break;} cur_match_len += 1; }
        if cur_match_len < TDEFL_MIN_MATCH_LEN {cur_match_len = 0;} else {cur_match_dist = 1;}
      }
    }
    else
    {
      let _borrow_temp_1 = d.m_lookahead_pos; let _borrow_temp_2 = d.m_dict_size; let _borrow_temp_3 = d.m_lookahead_size;
      tdefl_find_match(d, _borrow_temp_1, _borrow_temp_2, _borrow_temp_3, &mut cur_match_dist, &mut cur_match_len);
    }
    if ((cur_match_len == TDEFL_MIN_MATCH_LEN) && (cur_match_dist >= 8usize*1024)) || (cur_pos == cur_match_dist) || (d.m_flags.contains(TDEFL_FILTER_MATCHES) && (cur_match_len <= 5))
    {
      cur_match_len = 0;
      cur_match_dist = cur_match_len;
    }
    if d.m_saved_match_len > 0
    {
      if cur_match_len > d.m_saved_match_len
      {
        let _borrow_temp = d.m_saved_lit as u8;
        tdefl_record_literal(d, _borrow_temp);
        if cur_match_len >= 128
        {
          tdefl_record_match(d, cur_match_len, cur_match_dist);
          d.m_saved_match_len = 0; len_to_move = cur_match_len;
        }
        else
        {
          d.m_saved_lit = d.m_dict[cur_pos]; d.m_saved_match_dist = cur_match_dist; d.m_saved_match_len = cur_match_len;
        }
      }
      else
      {
        let _borrow_temp_1 = d.m_saved_match_len;
        let _borrow_temp_2 = d.m_saved_match_dist;
        tdefl_record_match(d, _borrow_temp_1, _borrow_temp_2);
        len_to_move = d.m_saved_match_len - 1; d.m_saved_match_len = 0;
      }
    }
    else if cur_match_dist == 0 {
      let _borrow_temp: u8 = d.m_dict[min(cur_pos, d.m_dict.size_of() - 1)];
      tdefl_record_literal(d, _borrow_temp);
    }
    else if (d.m_greedy_parsing) || d.m_flags.contains(TDEFL_RLE_MATCHES) || (cur_match_len >= 128)
    {
      tdefl_record_match(d, cur_match_len, cur_match_dist);
      len_to_move = cur_match_len;
    }
    else
    {
      d.m_saved_lit = d.m_dict[min(cur_pos, d.m_dict.size_of() - 1)]; d.m_saved_match_dist = cur_match_dist; d.m_saved_match_len = cur_match_len;
    }
    // Move the lookahead forward by len_to_move bytes.
    d.m_lookahead_pos += len_to_move;
    assert!(d.m_lookahead_size >= len_to_move);
    d.m_lookahead_size -= len_to_move;
    d.m_dict_size = min(d.m_dict_size + len_to_move, TDEFL_LZ_DICT_SIZE);
    // Check if it's time to flush the current LZ codes to the internal output buffer.
    if  (d.m_pLZ_code_buf as *const u8 > d.m_lz_code_buf.as_ptr().offset(TDEFL_LZ_CODE_BUF_SIZE as isize - 8)) ||
         ( (d.m_total_lz_bytes > 31*1024) && ((((((d.m_pLZ_code_buf as usize - d.m_lz_code_buf.as_ptr() as usize)) * 115) >> 7) >= d.m_total_lz_bytes) || d.m_flags.contains(TDEFL_FORCE_ALL_RAW_BLOCKS)))
    {
      let n: i8;
      d.m_pSrc = pSrc; d.m_src_buf_left = src_buf_left;
      if {n = tdefl_flush_block(d, TDEFL_NO_FLUSH) as i8; n != 0} {
        return if n < 0 {false} else {true};
      }
    }
  }

  d.m_pSrc = pSrc; d.m_src_buf_left = src_buf_left;
  return true;
}

unsafe fn tdefl_flush_output_buffer(d: &mut tdefl_compressor) -> tdefl_status
{
  if !d.m_pIn_buf_size.is_null()
  {
    *d.m_pIn_buf_size = d.m_pSrc as usize - d.m_pIn_buf as usize;
  }

  if !d.m_pOut_buf_size.is_null()
  {
    let n: usize = min(*d.m_pOut_buf_size - d.m_out_buf_ofs, d.m_output_flush_remaining);
    copy(d.m_output_buf.as_ptr().offset(d.m_output_flush_ofs as isize), d.m_pOut_buf.offset(d.m_out_buf_ofs as isize) as *mut u8, n);
    d.m_output_flush_ofs += n;
    d.m_output_flush_remaining -= n;
    d.m_out_buf_ofs += n;

    *d.m_pOut_buf_size = d.m_out_buf_ofs;
  }

  return if d.m_finished && d.m_output_flush_remaining==0 {TDEFL_STATUS_DONE} else {TDEFL_STATUS_OKAY};
}

/// Compresses a block of data, consuming as much of the specified input buffer as possible, and writing as much compressed data to the specified output buffer as possible.
unsafe fn tdefl_compress(d: &mut tdefl_compressor, pIn_buf: *const u8, pIn_buf_size: *mut usize, pOut_buf: *mut u8, pOut_buf_size: *mut usize, flush: tdefl_flush) -> tdefl_status
{
  /*if (!d)
  {
    if pIn_buf_size {*pIn_buf_size = 0;}
    if pOut_buf_size {*pOut_buf_size = 0;}
    return TDEFL_STATUS_BAD_PARAM;
  }*/

  d.m_pIn_buf = pIn_buf; d.m_pIn_buf_size = pIn_buf_size;
  d.m_pOut_buf = pOut_buf; d.m_pOut_buf_size = pOut_buf_size;
  d.m_pSrc = pIn_buf as *const u8; d.m_src_buf_left = if !pIn_buf_size.is_null() {*pIn_buf_size} else {0};
  d.m_out_buf_ofs = 0;
  d.m_flush = flush;

  if (d.m_pPut_buf_func.is_some() == (!pOut_buf.is_null() || !pOut_buf_size.is_null()))
      || (d.m_prev_return_status != TDEFL_STATUS_OKAY)
      || (d.m_wants_to_finish && (flush != TDEFL_FINISH))
      || (!pIn_buf_size.is_null() && *pIn_buf_size > 0 && pIn_buf.is_null())
      || (!pOut_buf_size.is_null() && *pOut_buf_size > 0 && pOut_buf.is_null())
  {
    if !pIn_buf_size.is_null() {*pIn_buf_size = 0;}
    if !pOut_buf_size.is_null() {*pOut_buf_size = 0;}
    d.m_prev_return_status = TDEFL_STATUS_BAD_PARAM;
    return TDEFL_STATUS_BAD_PARAM;
  }
  d.m_wants_to_finish |= flush == TDEFL_FINISH;

  if d.m_output_flush_remaining>0 || d.m_finished {
    d.m_prev_return_status = tdefl_flush_output_buffer(d);
    return d.m_prev_return_status;
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
  fn todo_name_me4(d: &tdefl_compressor) -> bool {
    ((d.m_flags & TDEFL_MAX_PROBES_MASK == TDEFL_FASTEST_MAX_PROBES) &&
     d.m_flags.contains(TDEFL_GREEDY_PARSING_FLAG) &&
     !d.m_flags.contains(TDEFL_FILTER_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS | TDEFL_RLE_MATCHES))
  }
  #[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little")))]
  fn todo_name_me4(d: &tdefl_compressor) -> bool { false }

// #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
  if todo_name_me4(d)
  {
    if !tdefl_compress_fast(d) {
      return d.m_prev_return_status;
    }
  }
  else
// #endif // #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
  {
    if !tdefl_compress_normal(d) {
      return d.m_prev_return_status;
    }
  }

  if d.m_flags.contains(TDEFL_WRITE_ZLIB_HEADER | TDEFL_COMPUTE_ADLER32) && !pIn_buf.is_null() {

    d.m_adler32 = mz_adler32(d.m_adler32,
      Some(from_raw_parts(pIn_buf as *const u8, d.m_pSrc as usize - pIn_buf as usize))); 
  }

  if flush!=TDEFL_NO_FLUSH && d.m_lookahead_size==0 && d.m_src_buf_left==0 && d.m_output_flush_remaining==0
  {
    if (tdefl_flush_block(d, flush) as i8) < 0 {
      return d.m_prev_return_status;
    }
    d.m_finished = flush == TDEFL_FINISH;
    if flush == TDEFL_FULL_FLUSH {
      for i in d.m_hash.iter_mut() {*i = 0u16;};
      for i in d.m_next.iter_mut() {*i = 0u16;};
      d.m_dict_size = 0;
    }
  }

  d.m_prev_return_status = tdefl_flush_output_buffer(d);
  return d.m_prev_return_status;
}

// tdefl_compress_mem_to_output() compresses a block to an output stream. The above helpers use this function internally.
fn tdefl_compress_mem_to_output(in_buf: &[u8], put_buf_func: tdefl_put_buf_func_ptr, flags: CompressionFlags) -> bool
{
  let mut comp = unsafe {tdefl_compressor::new(put_buf_func, flags)};
  let mut in_buf_size = in_buf.len();
  return unsafe {tdefl_compress(
      &mut comp,
      in_buf.as_ptr(),
      &mut in_buf_size as *mut usize,
      null::<u8>() as *mut u8,
      null::<usize>() as *mut usize,
      TDEFL_FINISH) == TDEFL_STATUS_DONE};
}

/// High level compression functions:
/// tdefl_compress_mem_to_heap() compresses a block in memory to a heap block allocated via malloc().
/// On entry:
///  pSrc_buf, src_buf_len: Pointer and size of source block to compress.
///  flags: The max match finder probes (default is 128) logically OR'd against the above flags. Higher probes are slower but improve compression.
/// On return:
///  Function returns a pointer to the compressed data, or NULL on failure.
///  *pOut_len will be set to the compressed data's size, which could be larger than src_buf_len on uncompressible data.
///  The caller must free() the returned block when it's no longer needed.
pub fn tdefl_compress_mem_to_heap(src_buf: &[u8], flags: CompressionFlags) -> Option<Vec<u8>>
{
  let mut out_buf = vec![0u8; 128];
  let mut out_buf_ofs = 0usize;

  if !tdefl_compress_mem_to_output(
    src_buf,
    &mut |pBuf: *const u8, len: usize| {
      let new_size = out_buf_ofs + len;
      let _borrow_temp = out_buf.len();
      if new_size > out_buf.len() {
        out_buf.extend(::std::iter::repeat(0u8).take(_borrow_temp));
      }
      unsafe {
        for (i,j) in out_buf[out_buf_ofs..new_size].iter_mut().zip(from_raw_parts(pBuf, len).iter()) {
          *i = *j;
        }
      };
      out_buf_ofs = new_size;
      true
    },
    flags) {return None;};

  unsafe { out_buf.set_len(out_buf_ofs) };
  return Some(out_buf);
}

/// tdefl_compress_mem_to_mem() compresses a block in memory to another block in memory.
/// Returns 0 on failure.
pub fn tdefl_compress_mem_to_mem(dst_buf: &mut[u8], src_buf: &[u8], flags: CompressionFlags) -> Option<usize>
{
  let mut out_size = 0usize;

  if !tdefl_compress_mem_to_output(
    src_buf,
    &mut |pBuf: *const u8, len: usize| {
      let new_size = out_size + len;
      if new_size > dst_buf.len() { return false; }
      unsafe {

        for (i,j) in dst_buf[out_size..new_size].iter_mut()
            .zip(from_raw_parts(pBuf, len).iter())
        {
          *i = *j;
        }

      };
      out_size = new_size;
      true
    },
    flags) {return None;};

  return Some(out_size);
}
