/* miniz.c v1.14 - public domain deflate/inflate, zlib-subset, ZIP reading/writing/appending, PNG writing
   See "unlicense" statement at the end of this file.
   Rich Geldreich <richgel99@gmail.com>, last updated May 20, 2012
   Implements RFC 1950: http://www.ietf.org/rfc/rfc1950.txt and RFC 1951: http://www.ietf.org/rfc/rfc1951.txt

   Most API's defined in miniz.c are optional. For example, to disable the archive related functions just define
   MINIZ_NO_ARCHIVE_APIS, or to get rid of all stdio usage define MINIZ_NO_STDIO (see the list below for more macros).

   * Change History
     5/20/12 v1.14 - MinGW32/64 GCC 4.6.1 compiler fixes: added MZ_FORCEINLINE, #include <time.h> (thanks fermtect).
     5/19/12 v1.13 - From jason@cornsyrup.org and kelwert@mtu.edu - Fix mz_crc32() so it doesn't compute the wrong CRC-32's when mz_ulong is 64-bit.
       Temporarily/locally slammed in "typedef unsigned long mz_ulong" and re-ran a randomized regression test on ~500k files.
       Eliminated a bunch of warnings when compiling with GCC 32-bit/64.
       Ran all examples, miniz.c, and tinfl.c through MSVC 2008's /analyze (static analysis) option and fixed all warnings (except for the silly
       "Use of the comma-operator in a tested expression.." analysis warning, which I purposely use to work around a MSVC compiler warning).
       Created 32-bit and 64-bit Codeblocks projects/workspace. Built and tested Linux executables. The codeblocks workspace is compatible with Linux+Win32/x64.
       Added miniz_tester solution/project, which is a useful little app derived from LZHAM's tester app that I use as part of the regression test.
       Ran miniz.c and tinfl.c through another series of regression testing on ~500,000 files and archives.
       Modified example5.c so it purposely disables a bunch of high-level functionality (MINIZ_NO_STDIO, etc.). (Thanks to corysama for the MINIZ_NO_STDIO bug report.)
       Fix ftell() usage in examples so they exit with an error on files which are too large (a limitation of the examples, not miniz itself).
     4/12/12 v1.12 - More comments, added low-level example5.c, fixed a couple minor level_and_flags issues in the archive API's.
      level_and_flags can now be set to MZ_DEFAULT_COMPRESSION. Thanks to Bruce Dawson <bruced@valvesoftware.com> for the feedback/bug report.
     5/28/11 v1.11 - Added statement from unlicense.org
     5/27/11 v1.10 - Substantial compressor optimizations:
      Level 1 is now ~4x faster than before. The L1 compressor's throughput now varies between 70-110MB/sec. on a
      Core i7 (actual throughput varies depending on the type of data, and x64 vs. x86).
      Improved baseline L2-L9 compression perf. Also, greatly improved compression perf. issues on some file types.
      Refactored the compression code for better readability and maintainability.
      Added level 10 compression level (L10 has slightly better ratio than level 9, but could have a potentially large
      drop in throughput on some files).
     5/15/11 v1.09 - Initial stable release.

   * Low-level Deflate/Inflate implementation notes:

     Compression: Use the "tdefl" API's. The compressor supports raw, static, and dynamic blocks, lazy or
     greedy parsing, match length filtering, RLE-only, and Huffman-only streams. It performs and compresses
     approximately as well as zlib.

     Decompression: Use the "tinfl" API's. The entire decompressor is implemented as a single function
     coroutine: see tinfl_decompress(). It supports decompression into a 32KB (or larger power of 2) wrapping buffer, or into a memory
     block large enough to hold the entire file.

     The low-level tdefl/tinfl API's do not make any use of dynamic memory allocation.

   * Important: For best perf. be sure to customize the below macros for your target platform:
     #define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 1
     #define MINIZ_LITTLE_ENDIAN 1
     #define MINIZ_HAS_64BIT_REGISTERS 1
*/

#![feature(macro_rules, slicing_syntax, globs, unboxed_closures)]
#![crate_type = "lib"]

extern crate libc;

use libc::{size_t, c_ulong, c_int, c_uchar, c_void};
use std::ptr::{copy_memory, set_memory, null};
use std::slice::raw::buf_as_slice;
use std::cmp::{max, min};
use memory_specific_constants::*;

trait SizeOf {
    fn size_of(&self) -> uint;
}

impl <T> SizeOf for T {
    fn size_of(&self) -> uint {
        use std::mem::{size_of};
        return size_of::<T>();
    }
}

// ------------------- zlib-style API Definitions.

// For more compatibility with zlib, miniz.c uses unsigned long for some parameters/struct members. Beware: mz_ulong can be either 32 or 64-bits!
type mz_ulong = libc::c_ulong;

const MZ_ADLER32_INIT: mz_ulong = (1);

const MZ_CRC32_INIT: mz_ulong = (0);

// Compression strategies.
enum CompressionStrategies { MZ_DEFAULT_STRATEGY = 0, MZ_FILTERED = 1, MZ_HUFFMAN_ONLY = 2, MZ_RLE = 3, MZ_FIXED = 4 }

// Method
const MZ_DEFLATED: int = 8;

// ------------------- Types and macros

type mz_uint = libc::c_uint;

// ------------------- Low-level Decompression API Definitions

// Decompression flags used by tinfl_decompress().
// TINFL_FLAG_PARSE_ZLIB_HEADER: If set, the input has a valid zlib header and ends with an adler32 checksum (it's a valid zlib stream). Otherwise, the input is a raw deflate stream.
// TINFL_FLAG_HAS_MORE_INPUT: If set, there are more input bytes available beyond the end of the supplied input buffer. If clear, the input buffer contains all remaining input.
// TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF: If set, the output buffer is large enough to hold the entire decompressed stream. If clear, the output buffer is at least the size of the dictionary (typically 32KB).
// TINFL_FLAG_COMPUTE_ADLER32: Force adler-32 checksum computation of the decompressed bytes.
bitflags! {
  flags DecompressionFlags: u32 {
    const TINFL_FLAG_PARSE_ZLIB_HEADER = 1,
    const TINFL_FLAG_HAS_MORE_INPUT = 2,
    const TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF = 4,
    const TINFL_FLAG_COMPUTE_ADLER32 = 8
  }
}

const TINFL_DECOMPRESS_MEM_TO_MEM_FAILED: uint = -1;

pub type tinfl_put_buf_func_ptr<'a> = FnMut<(&'a [u8],), bool> + 'a;

type tinfl_decompressor = tinfl_decompressor_tag;

// Max size of LZ dictionary.
const TINFL_LZ_DICT_SIZE: uint = 32768;

// Return status.
#[repr(i8)]
#[deriving(PartialEq)]
enum tinfl_status
{
  TINFL_STATUS_BAD_PARAM = -3,
  TINFL_STATUS_ADLER32_MISMATCH = -2,
  TINFL_STATUS_FAILED = -1,
  TINFL_STATUS_DONE = 0,
  TINFL_STATUS_NEEDS_MORE_INPUT = 1,
  TINFL_STATUS_HAS_MORE_OUTPUT = 2
}

// Initializes the decompressor to its initial state.
fn tinfl_init(r: &mut tinfl_decompressor) {
  r.m_state = 0;
}
fn tinfl_get_adler32(r: &tinfl_decompressor) -> u32 {
  r.m_check_adler32
}

// Internal/private bits follow.
const TINFL_MAX_HUFF_TABLES: uint = 3;
const TINFL_MAX_HUFF_SYMBOLS_0: uint = 288;
const TINFL_MAX_HUFF_SYMBOLS_1: uint = 32;
const TINFL_MAX_HUFF_SYMBOLS_2: uint = 19;
const TINFL_FAST_LOOKUP_BITS: uint = 10;
const TINFL_FAST_LOOKUP_SIZE: uint = 1 << TINFL_FAST_LOOKUP_BITS;

struct tinfl_huff_table
{
  m_code_size: [u8, ..TINFL_MAX_HUFF_SYMBOLS_0],
  m_look_up: [i16, ..TINFL_FAST_LOOKUP_SIZE],
  m_tree: [i16, ..TINFL_MAX_HUFF_SYMBOLS_0 * 2]
}

#[cfg(target_word_size = "64")]
  type tinfl_bit_buf_t = u64;
#[cfg(target_word_size = "64")]
  const TINFL_BITBUF_SIZE: size_t = (64);
#[cfg(not(target_word_size = "64"))]
  type tinfl_bit_buf_t = u32;
#[cfg(not(target_word_size = "64"))]
  const TINFL_BITBUF_SIZE: size_t = (32);

struct tinfl_decompressor_tag
{
  m_state: u32, m_num_bits: u32, m_zhdr0: u32, m_zhdr1: u32, m_z_adler32: u32, m_final: u32, m_type: u32,
  m_check_adler32: u32, m_dist: u32, m_counter: u32, m_num_extra: u32, m_table_sizes: [u32, ..TINFL_MAX_HUFF_TABLES],
  m_bit_buf: tinfl_bit_buf_t,
  m_dist_from_out_buf_start: libc::size_t,
  m_tables: [tinfl_huff_table, ..TINFL_MAX_HUFF_TABLES],
  m_raw_header: [u8, ..4], m_len_codes: [u8, ..TINFL_MAX_HUFF_SYMBOLS_0 + TINFL_MAX_HUFF_SYMBOLS_1 + 137]
}

// ------------------- Low-level Compression API Definitions

// Set TDEFL_LESS_MEMORY to 1 to use less memory (compression will be slightly slower, and raw/dynamic blocks will be output more frequently).
const TDEFL_LESS_MEMORY: int = 0;

// tdefl_init() compression flags logically OR'd together (low 12 bits contain the max. number of probes per dictionary search):
// TDEFL_DEFAULT_MAX_PROBES: The compressor defaults to 128 dictionary probes per dictionary search. 0=Huffman only, 1=Huffman+LZ (fastest/crap compression), 4095=Huffman+LZ (slowest/best compression).
enum InitCompressionFlags
{
  TDEFL_HUFFMAN_ONLY = 0, TDEFL_DEFAULT_MAX_PROBES = 128, TDEFL_MAX_PROBES_MASK = 0xFFF
}

// TDEFL_WRITE_ZLIB_HEADER: If set, the compressor outputs a zlib header before the deflate data, and the Adler-32 of the source data at the end. Otherwise, you'll get raw deflate data.
// TDEFL_COMPUTE_ADLER32: Always compute the adler-32 of the input data (even when not writing zlib headers).
// TDEFL_GREEDY_PARSING_FLAG: Set to use faster greedy parsing, instead of more efficient lazy parsing.
// TDEFL_NONDETERMINISTIC_PARSING_FLAG: Enable to decrease the compressor's initialization time to the minimum, but the output may vary from run to run given the same input (depending on the contents of memory).
// TDEFL_RLE_MATCHES: Only look for RLE matches (matches with a distance of 1)
// TDEFL_FILTER_MATCHES: Discards matches <= 5 chars if enabled.
// TDEFL_FORCE_ALL_STATIC_BLOCKS: Disable usage of optimized Huffman tables.
// TDEFL_FORCE_ALL_RAW_BLOCKS: Only use raw (uncompressed) deflate blocks.
enum OtherCompressionFlags
{
  TDEFL_WRITE_ZLIB_HEADER             = 0x01000,
  TDEFL_COMPUTE_ADLER32               = 0x02000,
  TDEFL_GREEDY_PARSING_FLAG           = 0x04000,
  TDEFL_NONDETERMINISTIC_PARSING_FLAG = 0x08000,
  TDEFL_RLE_MATCHES                   = 0x10000,
  TDEFL_FILTER_MATCHES                = 0x20000,
  TDEFL_FORCE_ALL_STATIC_BLOCKS       = 0x40000,
  TDEFL_FORCE_ALL_RAW_BLOCKS          = 0x80000
}

// Output stream interface. The compressor uses this interface to write compressed data. It'll typically be called TDEFL_OUT_BUF_SIZE at a time.
type tdefl_put_buf_func_ptr<'a> = &'a FnMut<(*const u8, uint), bool> + 'a;

const TDEFL_MAX_HUFF_TABLES: uint = 3;
const TDEFL_MAX_HUFF_SYMBOLS_0: uint = 288;
const TDEFL_MAX_HUFF_SYMBOLS_1: uint = 32;
const TDEFL_MAX_HUFF_SYMBOLS_2: uint = 19;
const TDEFL_MIN_MATCH_LEN: uint = 3;
const TDEFL_MAX_MATCH_LEN: uint = 258;
const TDEFL_LZ_DICT_SIZE: uint = 32768;
const TDEFL_LZ_DICT_SIZE_MASK: uint = (TDEFL_LZ_DICT_SIZE - 1);

// TDEFL_OUT_BUF_SIZE MUST be large enough to hold a single entire compressed output block (using static/fixed Huffman codes).
#[cfg(TDEFL_LESS_MEMORY)]
mod memory_specific_constants {
  pub const TDEFL_LZ_CODE_BUF_SIZE: uint = 24 * 1024;
  pub const TDEFL_OUT_BUF_SIZE: uint = (TDEFL_LZ_CODE_BUF_SIZE * 13 ) / 10;
  pub const TDEFL_MAX_HUFF_SYMBOLS: uint = 288;
  pub const TDEFL_LZ_HASH_BITS: uint = 12;
  pub enum TodoNameMeFlags { TDEFL_LEVEL1_HASH_SIZE_MASK = 4095, TDEFL_LZ_HASH_SIZE = 1 << TDEFL_LZ_HASH_BITS }
  pub const TDEFL_LZ_HASH_SHIFT: uint = (TDEFL_LZ_HASH_BITS + 2) / 3;
}
#[cfg(not(TDEFL_LESS_MEMORY))]
mod memory_specific_constants {
  pub const TDEFL_LZ_CODE_BUF_SIZE: uint = 64 * 1024;
  pub const TDEFL_OUT_BUF_SIZE: uint = (TDEFL_LZ_CODE_BUF_SIZE * 13 ) / 10;
  pub const TDEFL_MAX_HUFF_SYMBOLS: uint = 288;
  pub const TDEFL_LZ_HASH_BITS: uint = 15;
  pub enum TodoNameMeFlags { TDEFL_LEVEL1_HASH_SIZE_MASK = 4095, TDEFL_LZ_HASH_SIZE = 1 << TDEFL_LZ_HASH_BITS }
  pub const TDEFL_LZ_HASH_SHIFT: uint = (TDEFL_LZ_HASH_BITS + 2) / 3;
}

// The low-level tdefl functions below may be used directly if the above helper functions aren't flexible enough. The low-level functions don't make any heap allocations, unlike the above helper functions.
enum tdefl_status
{
  TDEFL_STATUS_BAD_PARAM = -2,
  TDEFL_STATUS_PUT_BUF_FAILED = -1,
  TDEFL_STATUS_OKAY = 0,
  TDEFL_STATUS_DONE = 1,
}

// Must map to MZ_NO_FLUSH, MZ_SYNC_FLUSH, etc. enums
enum tdefl_flush
{
  TDEFL_NO_FLUSH = 0,
  TDEFL_SYNC_FLUSH = 2,
  TDEFL_FULL_FLUSH = 3,
  TDEFL_FINISH = 4
}

// tdefl's compression state structure.
struct tdefl_compressor <'a>
{
  m_pPut_buf_func: tdefl_put_buf_func_ptr<'a>,
  m_pPut_buf_user: *mut c_void,
  m_flags: mz_uint, m_max_probes: [mz_uint, ..2],
  m_greedy_parsing: int,
  m_adler32: mz_uint, m_lookahead_pos: uint, m_lookahead_size: uint, m_dict_size: uint,
  m_pLZ_code_buf: *mut u8, m_pLZ_flags: *mut u8, m_pOutput_buf: *mut u8, m_pOutput_buf_end: *mut u8,
  m_num_flags_left: mz_uint, m_total_lz_bytes: uint, m_lz_code_buf_dict_pos: uint, m_bits_in: uint, m_bit_buffer: uint,
  m_saved_match_dist: mz_uint, m_saved_match_len: mz_uint, m_saved_lit: mz_uint, m_output_flush_ofs: uint, m_output_flush_remaining: mz_uint, m_finished: mz_uint, m_block_index: mz_uint, m_wants_to_finish: mz_uint,
  m_prev_return_status: tdefl_status,
  m_pIn_buf: *const libc::c_void,
  m_pOut_buf: *mut c_void,
  m_pIn_buf_size: *mut size_t, m_pOut_buf_size: *mut size_t,
  m_flush: tdefl_flush,
  m_pSrc: *const u8,
  m_src_buf_left: size_t, m_out_buf_ofs: size_t,
  m_dict: [u8, ..TDEFL_LZ_DICT_SIZE + TDEFL_MAX_MATCH_LEN - 1],
  // TODO verify this is the right order of indexes
  m_huff_count: [[u16, ..TDEFL_MAX_HUFF_SYMBOLS], ..TDEFL_MAX_HUFF_TABLES],
  m_huff_codes: [[u16, ..TDEFL_MAX_HUFF_SYMBOLS], ..TDEFL_MAX_HUFF_TABLES],
  m_huff_code_sizes: [[u8, ..TDEFL_MAX_HUFF_SYMBOLS], ..TDEFL_MAX_HUFF_TABLES],
  m_lz_code_buf: [u8, ..TDEFL_LZ_CODE_BUF_SIZE],
  m_next: [u16, ..TDEFL_LZ_DICT_SIZE],
  m_hash: [u16, ..TDEFL_LZ_HASH_SIZE],
  m_output_buf: [u8, ..TDEFL_OUT_BUF_SIZE],
}

// ------------------- End of Header: Implementation follows. (If you only want the header, define MINIZ_HEADER_FILE_ONLY.)

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
  macro_rules! MZ_READ_LE16( (p) => (*((p) as *const u16)); )
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
  macro_rules! MZ_READ_LE32( (p) => (*((p) as *const u32)); )

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little")))]
  macro_rules! MZ_READ_LE16( (p) => ((u32)(((p) as *const u8)[0]) | ((u32)(((p) as *const u8)[1]) << 8u)) )
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little")))]
  macro_rules! MZ_READ_LE32( (p) => ((u32)(((p) as *const u8)[0]) | ((u32)(((p) as *const u8)[1]) << 8u) | ((u32)(((p) as *const u8)[2]) << 16u) | ((u32)(((p) as *const u8)[3]) << 24u)) )

// ------------------- zlib-style API's

/// Adler-32 checksum algorithm
/// mz_adler32() returns the initial adler-32 value to use when called with ptr==NULL.
fn mz_adler32(adler: mz_ulong, buf: Option<&[u8]>) -> mz_ulong
{
  let mut buf: &[u8] = match buf { Some(x) => x, None => return MZ_ADLER32_INIT };
  let mut s1: u32 = (adler & 0xffff) as u32;
  let mut s2: u32 = (adler >> 16) as u32;
  let mut block_len: uint = (buf.len() % 5552u);
  while buf.len() > 0 {
    for i in buf[..block_len].iter() {
      s1 += (*i as u32); s2 += s1;
    }
    s1 %= 65521u32; s2 %= 65521u32; buf = buf[block_len..]; block_len = 5552u;
  }
  return ((s2 << 16) + s1) as mz_ulong;
}

/// Karl Malbrain's compact CRC-32. See "A compact CCITT crc16 and crc32 C implementation that balances processor cache usage against speed": http://www.geocities.com/malbrain/
/// mz_crc32() returns the initial CRC-32 value to use when called with ptr==NULL.
fn mz_crc32(crc: mz_ulong, buf: Option<&[u8]>) -> mz_ulong
{
  let mut buf: &[u8] = match buf { Some(x) => x, None => return MZ_CRC32_INIT };
  let s_crc32: [u32, ..16] = [ 0, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4, 0x4db26158, 0x5005713c,
    0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c, 0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c ];
  let crcu32: u32 = crc as u32;
  crcu32 = !crcu32;
  for b in buf.iter() {
    crcu32 = (crcu32 >> 4) ^ s_crc32[((crcu32 as u8 & 0xF) ^ (*b & 0xF)) as uint];
    crcu32 = (crcu32 >> 4) ^ s_crc32[((crcu32 as u8 & 0xF) ^ (*b >> 4)) as uint];
  }
  return !crcu32 as mz_ulong;
}

// ------------------- Low-level Decompression (completely independent from all compression API's)

macro_rules! TINFL_MEMCPY(($d:expr, $s:expr, $l:expr) => (memcpy(d, s, l););)
macro_rules! TINFL_MEMSET(($p:expr, $c:expr, $l:expr) => (memset(p, c, l););)

// #define TINFL_CR_BEGIN switch(r->m_state) { case 0:
macro_rules! TINFL_CR_RETURN( ($state_index: expr, $result: expr) => ( { status = result; r->m_state = state_index; goto common_exit; case state_index:; }; ); )
macro_rules! TINFL_CR_RETURN_FOREVER( ($state_index:expr, $result:expr) => ( { loop { TINFL_CR_RETURN(state_index, result); } }; ); )
// #define TINFL_CR_FINISH }

// TODO: If the caller has indicated that there's no more input, and we attempt to read beyond the input buf, then something is wrong with the input because the inflator never
// reads ahead more than it needs to. Currently TINFL_GET_BYTE() pads the end of the stream with 0's in this scenario.
macro_rules! TINFL_GET_BYTE( ($state_index:expr, $c:expr) => (loop {
  if (pIn_buf_cur >= pIn_buf_end) {
    loop {
      if (decomp_flags & TINFL_FLAG_HAS_MORE_INPUT) {
        TINFL_CR_RETURN(state_index, TINFL_STATUS_NEEDS_MORE_INPUT);
        if (pIn_buf_cur < pIn_buf_end) {
          c = *pIn_buf_cur;
          pIn_buf_cur += 1;
          break;
        }
      } else {
        c = 0;
        break;
      }
    }
  } else {c = *pIn_buf_cur; pIn_buf_cur += 1;} break; }; );
)

macro_rules! TINFL_NEED_BITS( ($state_index:expr, $n:expr) => (do { mz_uint c; TINFL_GET_BYTE!(state_index, c); bit_buf |= (((tinfl_bit_buf_t)c) << num_bits); num_bits += 8; } while (num_bits < (mz_uint)(n))); )
macro_rules! TINFL_SKIP_BITS( ($state_index:expr, $n:expr) => (do { if (num_bits < (mz_uint)(n)) { TINFL_NEED_BITS!(state_index, n); } bit_buf >>= (n); num_bits -= (n); } MZ_MACRO_END); )
macro_rules! TINFL_GET_BITS( ($state_index:expr, $b:expr, $n:expr) => (
  do { if (num_bits < (mz_uint)(n)) { TINFL_NEED_BITS!(state_index, n); } b = bit_buf & ((1 << (n)) - 1); bit_buf >>= (n); num_bits -= (n); } MZ_MACRO_END);
)

// TINFL_HUFF_BITBUF_FILL() is only used rarely, when the number of bytes remaining in the input buffer falls below 2.
// It reads just enough bytes from the input stream that are needed to decode the next Huffman code (and absolutely no more). It works by trying to fully decode a
// Huffman code by using whatever bits are currently present in the bit buffer. If this fails, it reads another byte, and tries again until it succeeds or until the
// bit buffer contains >=15 bits (deflate's max. Huffman code size).
macro_rules! TINFL_HUFF_BITBUF_FILL( (
  $state_index:expr, $pHuff:expr) => (
  loop {
    temp = (pHuff)->m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)];
    if (temp >= 0) {
      code_len = temp >> 9;
      if ((code_len) && (num_bits >= code_len))
      break;
    } else if (num_bits > TINFL_FAST_LOOKUP_BITS) {
       code_len = TINFL_FAST_LOOKUP_BITS;
       do {
          temp = (pHuff)->m_tree[~temp + ((bit_buf >> code_len++) & 1)];
       } while ((temp < 0) && (num_bits >= (code_len + 1))); if (temp >= 0) break;
    } TINFL_GET_BYTE!(state_index, c); bit_buf |= (((tinfl_bit_buf_t)c) << num_bits); num_bits += 8;
    if !(num_bits < 15) {break;};
  });
)

// TINFL_HUFF_DECODE() decodes the next Huffman coded symbol. It's more complex than you would initially expect because the zlib API expects the decompressor to never read
// beyond the final byte of the deflate stream. (In other words, when this macro wants to read another byte from the input, it REALLY needs another byte in order to fully
// decode the next Huffman code.) Handling this properly is particularly important on raw deflate (non-zlib) streams, which aren't followed by a byte aligned adler-32.
// The slow path is only executed at the very end of the input buffer.
macro_rules! TINFL_HUFF_DECODE( ($state_index:expr, $sym:expr, $pHuff:expr) => (loop {
  int temp; mz_uint code_len, c;
  if (num_bits < 15) {
    if ((pIn_buf_end - pIn_buf_cur) < 2) {
       TINFL_HUFF_BITBUF_FILL(state_index, pHuff);
    } else {
       bit_buf |= (((tinfl_bit_buf_t)pIn_buf_cur[0]) << num_bits) | (((tinfl_bit_buf_t)pIn_buf_cur[1]) << (num_bits + 8)); pIn_buf_cur += 2; num_bits += 16;
    }
  }
  if ((temp = (pHuff)->m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >= 0)
    code_len = temp >> 9, temp &= 511;
  else {
    code_len = TINFL_FAST_LOOKUP_BITS; do { temp = (pHuff)->m_tree[~temp + ((bit_buf >> code_len++) & 1)]; } while (temp < 0);
  } sym = temp; bit_buf >>= code_len; num_bits -= code_len; break;}; );
)

// Main low-level decompressor coroutine function. This is the only function actually needed for decompression. All the other functions are just high-level helpers for improved usability.
// This is a universal API, i.e. it can be used as a building block to build any desired higher level decompression API. In the limit case, it can be called once per every byte input or output.
fn tinfl_decompress(r: &mut tinfl_decompressor, pIn_buf_next: *const u8, pIn_buf_size: &mut uint, pOut_buf_start: *const u8, pOut_buf_next: *const u8, pOut_buf_size: &mut uint, decomp_flags: DecompressionFlags) -> tinfl_status
{
  let s_length_base: [int, ..31] = [ 3,4,5,6,7,8,9,10,11,13, 15,17,19,23,27,31,35,43,51,59, 67,83,99,115,131,163,195,227,258,0,0 ];
  let s_length_extra: [int, ..31]= [ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0 ];
  let s_dist_base: [int, ..32] = [ 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193, 257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0];
  let s_dist_extra: [int, ..32] = [ 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13, 0xFFFF, 0xFFFF ]; // WARNING miniz.c HAD WRONG INITIALIZER
  let s_length_dezigzag: [u8, ..19] = [ 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 ];
  let s_min_table_sizes: [int, ..3] = [ 257, 1, 4 ];

  let status: tinfl_status = TINFL_STATUS_FAILED; let num_bits: u32; let dist: u32; let counter: u32; let num_extra: u32; let bit_buf: tinfl_bit_buf_t;
  let pIn_buf_cur: *const u8 = pIn_buf_next; let pIn_buf_end: *const u8 = pIn_buf_next.offset(*pIn_buf_size as int);
  let pOut_buf_cur: *const u8 = pOut_buf_next; let pOut_buf_end:  *const u8 = pOut_buf_next.offset(*pOut_buf_size as int);
  let out_buf_size_mask: size_t = if decomp_flags.contains(TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF) {-1 as size_t} else {((pOut_buf_next as uint - pOut_buf_start as uint) + *pOut_buf_size) as u64 - 1};
  let dist_from_out_buf_start: size_t;

  // Ensure the output buffer's size is a power of 2, unless the output buffer is large enough to hold the entire output file (in which case it doesn't matter).
  if (((out_buf_size_mask + 1) & out_buf_size_mask != 0) || (pOut_buf_next < pOut_buf_start)) { *pIn_buf_size = 0; *pOut_buf_size = 0; return TINFL_STATUS_BAD_PARAM; }

  num_bits = r.m_num_bits; bit_buf = r.m_bit_buf; dist = r.m_dist; counter = r.m_counter; num_extra = r.m_num_extra; dist_from_out_buf_start = r.m_dist_from_out_buf_start;
  // TINFL_CR_BEGIN
  {
    /*
    bit_buf = num_bits = dist = counter = num_extra = r.m_zhdr0 = r.m_zhdr1 = 0; r.m_z_adler32 = r.m_check_adler32 = 1;
    if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER)
    {
      TINFL_GET_BYTE!(1, r.m_zhdr0); TINFL_GET_BYTE!(2, r.m_zhdr1);
      counter = (((r.m_zhdr0 * 256 + r.m_zhdr1) % 31 != 0) || (r.m_zhdr1 & 32) || ((r.m_zhdr0 & 15) != 8));
      if (!(decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF)) {counter |= (((1u << (8u + (r.m_zhdr0 >> 4))) > 32768u) || ((out_buf_size_mask + 1) < (size_t)(1u << (8u + (r.m_zhdr0 >> 4)))))};
      if (counter) { TINFL_CR_RETURN_FOREVER(36, TINFL_STATUS_FAILED); }
    }
    loop
    {
      TINFL_GET_BITS(3, r.m_final, 3); r.m_type = r.m_final >> 1;
      if (r.m_type == 0)
      {
        TINFL_SKIP_BITS(5, num_bits & 7);
        counter = 0;
        while counter < 4 { if (num_bits) {TINFL_GET_BITS(6, r.m_raw_header[counter], 8);} else {TINFL_GET_BYTE(7, r.m_raw_header[counter]);}; counter+=1}
        if ((counter = (r.m_raw_header[0] | (r.m_raw_header[1] << 8))) != (mz_uint)(0xFFFF ^ (r.m_raw_header[2] | (r.m_raw_header[3] << 8)))) { TINFL_CR_RETURN_FOREVER(39, TINFL_STATUS_FAILED); }
        while ((counter) && (num_bits))
        {
          TINFL_GET_BITS(51, dist, 8);
          while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN(52, TINFL_STATUS_HAS_MORE_OUTPUT); }
          *pOut_buf_cur = dist as u8; pOut_buf_cur+=1;
          counter-=1;
        }
        while (counter)
        {
          let n: size_t; while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN(9, TINFL_STATUS_HAS_MORE_OUTPUT); }
          while (pIn_buf_cur >= pIn_buf_end)
          {
            if (decomp_flags & TINFL_FLAG_HAS_MORE_INPUT)
            {
              TINFL_CR_RETURN(38, TINFL_STATUS_NEEDS_MORE_INPUT);
            }
            else
            {
              TINFL_CR_RETURN_FOREVER(40, TINFL_STATUS_FAILED);
            }
          }
          n = MZ_MIN(MZ_MIN((size_t)(pOut_buf_end - pOut_buf_cur), (size_t)(pIn_buf_end - pIn_buf_cur)), counter);
          TINFL_MEMCPY(pOut_buf_cur, pIn_buf_cur, n); pIn_buf_cur += n; pOut_buf_cur += n; counter -= n as mz_uint;
        }
      }
      else if (r.m_type == 3)
      {
        TINFL_CR_RETURN_FOREVER(10, TINFL_STATUS_FAILED);
      }
      else
      {
        if (r.m_type == 1)
        {
          mz_uint8 *p = r.m_tables[0].m_code_size; let i: mz_uint;
          r.m_table_sizes[0] = 288; r.m_table_sizes[1] = 32; TINFL_MEMSET(r.m_tables[1].m_code_size, 5, 32);
          i = 0;
          while i <= 143 {*p = 8; p+=1; i+=1;}
          while i <= 255 {*p = 9; p+=1; i+=1;}
          while i <= 279 {*p = 7; p+=1; i+=1;}
          while i <= 287 {*p = 8; p+=1; i+=1;}
        }
        else
        {
          counter = 0;
          while counter < 3 { TINFL_GET_BITS(11, r.m_table_sizes[counter], "\05\05\04"[counter]); r.m_table_sizes[counter] += s_min_table_sizes[counter]; counter+=1; }
          MZ_CLEAR_OBJ(r.m_tables[2].m_code_size); counter = 0; while counter < r.m_table_sizes[2] { let s: mz_uint; TINFL_GET_BITS(14, s, 3); r.m_tables[2].m_code_size[s_length_dezigzag[counter]] = s as mz_uint8; counter+=1; }
          r.m_table_sizes[2] = 19;
        }
        while r.m_type as int >= 0
        {
          let tree_next: int; let tree_cur: int; let pTable: *mut tinfl_huff_table;
          let i: mz_uint; let j: mz_uint; let used_syms: mz_uint; let total: mz_uint; let sym_index: mz_uint;
          let next_code: [mz_uint, ..17]; let total_syms: [mz_uint, ..16]; pTable = &r.m_tables[r.m_type]; MZ_CLEAR_OBJ(total_syms); MZ_CLEAR_OBJ(pTable.m_look_up); MZ_CLEAR_OBJ(pTable.m_tree);
          i = 0;
          while i < r.m_table_sizes[r.m_type] {total_syms[pTable.m_code_size[i]] += 1; i += 1;}
          used_syms = 0; total = 0; next_code[0] = next_code[1] = 0;
          i = 1;
          while i <= 15 { used_syms += total_syms[i]; next_code[i + 1] = (total = ((total + total_syms[i]) << 1)); i += 1; }
          if ((65536 != total) && (used_syms > 1))
          {
            TINFL_CR_RETURN_FOREVER(35, TINFL_STATUS_FAILED);
          }
          tree_next = -1; sym_index = 0;
          while sym_index < r.m_table_sizes[r.m_type]
          {
            let rev_code: mz_uint = 0;
            let l: mz_uint;
            let cur_code: mz_uint;
            let code_size: mz_uint = pTable.m_code_size[sym_index]; if !code_size {continue;}
            cur_code = next_code[code_size]+=1;
            l = code_size; while l > 0 {rev_code = (rev_code << 1) | (cur_code & 1); l-=1; cur_code >>= 1;}
            if (code_size <= TINFL_FAST_LOOKUP_BITS) {
              let k: i16 = ((code_size << 9) | sym_index) as i16;
              while rev_code < TINFL_FAST_LOOKUP_SIZE { pTable.m_look_up[rev_code] = k; rev_code += (1 << code_size); }
              continue;
            }
            if (0 == (tree_cur = pTable.m_look_up[rev_code & (TINFL_FAST_LOOKUP_SIZE - 1)])) {
              pTable.m_look_up[rev_code & (TINFL_FAST_LOOKUP_SIZE - 1)] = tree_next as i16;
              tree_cur = tree_next;
              tree_next -= 2;
            }
            rev_code >>= (TINFL_FAST_LOOKUP_BITS - 1);
            j = code_size;
            while j > (TINFL_FAST_LOOKUP_BITS + 1)
            {
              tree_cur -= ((rev_code >>= 1) & 1);
              if (!pTable.m_tree[-tree_cur - 1]) { pTable.m_tree[-tree_cur - 1] = tree_next as i16; tree_cur = tree_next; tree_next -= 2; } else {tree_cur = pTable.m_tree[-tree_cur - 1];}
              j -= 1;
            }
            tree_cur -= ((rev_code >>= 1) & 1); pTable.m_tree[-tree_cur - 1] = sym_index as i16;
            sym_index += 1;
          }
          if (r.m_type == 2)
          {
            counter = 0;
            while counter < (r.m_table_sizes[0] + r.m_table_sizes[1])
            {
              let s: mz_uint;
              TINFL_HUFF_DECODE(16, dist, &r.m_tables[2]); if (dist < 16) { r.m_len_codes[counter] = dist as u8; counter+=1; continue; }
              if ((dist == 16) && (!counter))
              {
                TINFL_CR_RETURN_FOREVER(17, TINFL_STATUS_FAILED);
              }
              num_extra = "\02\03\07"[dist - 16]; TINFL_GET_BITS(18, s, num_extra); s += "\03\03\013"[dist - 16];
              TINFL_MEMSET(r.m_len_codes + counter, if dist == 16 {r.m_len_codes[counter - 1]} else {0}, s); counter += s;
            }
            if ((r.m_table_sizes[0] + r.m_table_sizes[1]) != counter)
            {
              TINFL_CR_RETURN_FOREVER(21, TINFL_STATUS_FAILED);
            }
            TINFL_MEMCPY(r.m_tables[0].m_code_size, r.m_len_codes, r.m_table_sizes[0]); TINFL_MEMCPY(r.m_tables[1].m_code_size, r.m_len_codes + r.m_table_sizes[0], r.m_table_sizes[1]);
          }
          r.m_type-=1;
        }
        loop
        {
          let pSrc: *const u8;
          loop
          {
            if (((pIn_buf_end - pIn_buf_cur) < 4) || ((pOut_buf_end - pOut_buf_cur) < 2))
            {
              TINFL_HUFF_DECODE(23, counter, &r.m_tables[0]);
              if (counter >= 256){
                break;
              }
              while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN(24, TINFL_STATUS_HAS_MORE_OUTPUT); }
              *pOut_buf_cur = counter as u8;
              pOut_buf_cur += 1;
            }
            else
            {
              let sym2: c_int; let code_len: mz_uint;

              #[cfg(target_word_size = "64")]
              fn todo_name_me() {if (num_bits < 30) { bit_buf |= ((MZ_READ_LE32(pIn_buf_cur) as tinfl_bit_buf_t) << num_bits); pIn_buf_cur += 4; num_bits += 32; }}
              #[cfg(not(target_word_size = "64"))]
              fn todo_name_me() {if (num_bits < 15) { bit_buf |= ((MZ_READ_LE16(pIn_buf_cur) as tinfl_bit_buf_t) << num_bits); pIn_buf_cur += 2; num_bits += 16; }}
              todo_name_me();

              if ((sym2 = r.m_tables[0].m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >= 0){
                code_len = sym2 >> 9;
              } else {
                code_len = TINFL_FAST_LOOKUP_BITS; loop { sym2 = r.m_tables[0].m_tree[!sym2 + ((bit_buf >> code_len) & 1)]; code_len += 1; if !(sym2 < 0) {break;} };
              }
              counter = sym2; bit_buf >>= code_len; num_bits -= code_len;
              if (counter & 256){
                break;
              }

              #[cfg(target_word_size = "64")]
              fn todo_name_me2() {if (num_bits < 15) { bit_buf |= ((MZ_READ_LE16(pIn_buf_cur) as tinfl_bit_buf_t) << num_bits); pIn_buf_cur += 2; num_bits += 16; }}
              #[cfg(not(target_word_size = "64"))]
              fn todo_name_me2() {}
              todo_name_me2();

              if ((sym2 = r.m_tables[0].m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >= 0){
                code_len = sym2 >> 9;
              } else {
                code_len = TINFL_FAST_LOOKUP_BITS; loop { sym2 = r.m_tables[0].m_tree[!sym2 + ((bit_buf >> code_len) & 1)]; code_len += 1; if !(sym2 < 0) {break;} };
              }
              bit_buf >>= code_len; num_bits -= code_len;

              pOut_buf_cur[0] = counter as u8;
              if (sym2 & 256)
              {
                pOut_buf_cur+=1;
                counter = sym2;
                break;
              }
              pOut_buf_cur[1] = sym2 as u8;
              pOut_buf_cur += 2;
            }
          }
          if ((counter &= 511) == 256) {break;}

          num_extra = s_length_extra[counter - 257]; counter = s_length_base[counter - 257];
          if num_extra { let extra_bits: mz_uint; TINFL_GET_BITS(25, extra_bits, num_extra); counter += extra_bits; }

          TINFL_HUFF_DECODE(26, dist, &r.m_tables[1]);
          num_extra = s_dist_extra[dist]; dist = s_dist_base[dist];
          if num_extra { let extra_bits: mz_uint; TINFL_GET_BITS(27, extra_bits, num_extra); dist += extra_bits; }

          dist_from_out_buf_start = pOut_buf_cur - pOut_buf_start;
          if ((dist > dist_from_out_buf_start) && (decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF))
          {
            TINFL_CR_RETURN_FOREVER(37, TINFL_STATUS_FAILED);
          }

          pSrc = pOut_buf_start + ((dist_from_out_buf_start - dist) & out_buf_size_mask);

          if ((MZ_MAX(pOut_buf_cur, pSrc) + counter) > pOut_buf_end)
          {
            while {let temp = counter; counter-=1; temp}
            {
              while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN(53, TINFL_STATUS_HAS_MORE_OUTPUT); }
              *pOut_buf_cur = pOut_buf_start[(dist_from_out_buf_start - dist) & out_buf_size_mask];
              dist_from_out_buf_start += 1;
              pOut_buf_cur += 1;
            }
            continue;
          }
          else if ((counter >= 9) && (counter <= dist))
          {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fn todo_name_me3() {
              let pSrc_end: *const u8 = pSrc + (counter & !7);
              loop
              {
                (pOut_buf_cur as *mut u32)[0] = (pSrc as *const u32)[0];
                (pOut_buf_cur as *mut u32)[1] = (pSrc as *const u32)[1];
                pOut_buf_cur += 8;
                if !((pSrc += 8) < pSrc_end) {break;}
              }
              if ((counter &= 7) < 3)
              {
                if (counter)
                {
                  pOut_buf_cur[0] = pSrc[0];
                  if (counter > 1){
                    pOut_buf_cur[1] = pSrc[1];
                  }
                  pOut_buf_cur += counter;
                }
                continue;
              }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            fn todo_name_me3() {}
            todo_name_me3()
          }
          loop
          {
            pOut_buf_cur[0] = pSrc[0];
            pOut_buf_cur[1] = pSrc[1];
            pOut_buf_cur[2] = pSrc[2];
            pOut_buf_cur += 3; pSrc += 3;
            if !((counter -= 3) as int > 2) {break;}
          }
          if (counter as int > 0)
          {
            pOut_buf_cur[0] = pSrc[0];
            if (counter as int > 1){
              pOut_buf_cur[1] = pSrc[1];
            }
            pOut_buf_cur += counter;
          }
        }
      }
      if (r.m_final & 1) as bool {break;}
    }
    if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER)
    {
      TINFL_SKIP_BITS(32, num_bits & 7); counter = 0; while counter < 4 { let s: mz_uint; if (num_bits) {TINFL_GET_BITS(41, s, 8);} else {TINFL_GET_BYTE(42, s);}; r.m_z_adler32 = (r.m_z_adler32 << 8) | s; counter+=1; }
    }
    TINFL_CR_RETURN_FOREVER(34, TINFL_STATUS_DONE); */
  // TINFL_CR_FINISH
  }

  let common_exit = || {
    r.m_num_bits = num_bits; r.m_bit_buf = bit_buf; r.m_dist = dist; r.m_counter = counter; r.m_num_extra = num_extra; r.m_dist_from_out_buf_start = dist_from_out_buf_start;
    *pIn_buf_size = pIn_buf_cur as uint - pIn_buf_next as uint;
    *pOut_buf_size = pOut_buf_cur as uint - pOut_buf_next as uint;
    if decomp_flags.contains(TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_COMPUTE_ADLER32) && (status as i8 >= 0)
    {
      buf_as_slice(pOut_buf_next, *pOut_buf_size, |out_buf_next_slice| {
        r.m_check_adler32 = mz_adler32(r.m_check_adler32 as mz_ulong, Some(out_buf_next_slice)) as u32;
      });
      if ((status == TINFL_STATUS_DONE) && decomp_flags.contains(TINFL_FLAG_PARSE_ZLIB_HEADER) && (r.m_check_adler32 != r.m_z_adler32)) {status = TINFL_STATUS_ADLER32_MISMATCH;};
    }
  };
  return status;
}

/// High level decompression functions:
/// tinfl_decompress_mem_to_heap() decompresses a block in memory to a heap block allocated via malloc().
/// On entry:
///  pSrc_buf, src_buf_len: Pointer and size of the Deflate or zlib source data to decompress.
/// On return:
///  Function returns a pointer to the decompressed data, or NULL on failure.
///  *pOut_len will be set to the decompressed data's size, which could be larger than src_buf_len on uncompressible data.
///  The caller must call mz_free() on the returned block when it's no longer needed.
pub fn tinfl_decompress_mem_to_heap(src_buf: &[u8], flags: DecompressionFlags) -> Option<Vec<u8>>
{
  let mut decomp: tinfl_decompressor;
  tinfl_init(&decomp);
  // Create output buffer.
  // WARNING: we no longer pass a NULL pointer to tinfl_decompress
  // on the first pass (as miniz.c did). TODO: ensure this decompresses correctly.
  let mut out_buf = Vec::from_elem(128, 0u8);
  // Track position in input buffer
  let mut src_buf_ofs: uint = 0;
  // Track position in output buffer
  let mut out_buf_ofs: uint = 0;
  loop {
    let mut src_buf_size: uint = src_buf.len() - src_buf_ofs;
    let mut out_buf_size: uint = out_buf.len() - out_buf_ofs;
    let status: tinfl_status = tinfl_decompress(
      &mut decomp,
      src_buf[src_buf_ofs..].as_ptr(),
      &mut src_buf_size,
      out_buf[].as_ptr(),
      out_buf[out_buf_ofs..].as_ptr(),
      &mut out_buf_size,
      (flags & !TINFL_FLAG_HAS_MORE_INPUT) | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);

    if (status as i8) < 0 || (status == TINFL_STATUS_NEEDS_MORE_INPUT) {return None;}
    // now, status is either TINFL_STATUS_HAS_MORE_OUTPUT or TINFL_STATUS_DONE

    // Increase buffer offsets to reflect newly copied data.
    // tinfl_decompress writes this back into src_buf_size and out_buf_size
    src_buf_ofs += src_buf_size;
    out_buf_ofs += out_buf_size;

    // If all data is copied, end.
    if status == TINFL_STATUS_DONE {break;}
    // Otherwise, double the output buffer capacity & length.
    out_buf.grow(out_buf.len(), 0u8);
  }
  // Set length of output buffer to number of bytes copied, instead of capacity.
  unsafe {out_buf.set_len(out_buf_ofs)};
  Some(out_buf)
}

// tinfl_decompress_mem_to_mem() decompresses a block in memory to another block in memory.
// Returns TINFL_DECOMPRESS_MEM_TO_MEM_FAILED on failure, or the number of bytes written on success.
pub fn tinfl_decompress_mem_to_mem(out_buf: &mut[u8], src_buf: &[u8], flags: DecompressionFlags) -> uint
{
  let mut decomp: tinfl_decompressor; tinfl_init(&decomp);
  let mut src_buf_len: uint = src_buf.len();
  let mut out_buf_len: uint = out_buf.len();
  let status: tinfl_status = tinfl_decompress(
    &mut decomp,
    src_buf.as_ptr(),
    &mut src_buf_len,
    out_buf.as_ptr(),
    out_buf.as_ptr(),
    &mut out_buf_len,
    (flags & !TINFL_FLAG_HAS_MORE_INPUT) | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
  if status != TINFL_STATUS_DONE {TINFL_DECOMPRESS_MEM_TO_MEM_FAILED} else {out_buf_len};
}

// tinfl_decompress_mem_to_callback() decompresses a block in memory to an internal 32KB buffer, and a user provided callback function will be called to flush the buffer.
// Returns 1 on success or 0 on failure.
pub fn tinfl_decompress_mem_to_callback(in_buf: &[u8], put_buf_func:&mut tinfl_put_buf_func_ptr, pPut_buf_user: *const c_void, flags: DecompressionFlags) -> (bool, uint)
{
  let decomp: tinfl_decompressor; tinfl_init(&decomp);
  let dict: Vec<u8> = Vec::from_elem(TINFL_LZ_DICT_SIZE, 0u8);
  let in_buf_ofs: uint = 0;
  let dict_ofs: uint = 0;
  let result: bool = false;
  loop {
    let in_buf_size: uint = in_buf.len() - in_buf_ofs;
    let dst_buf_size: uint = TINFL_LZ_DICT_SIZE - dict_ofs;
    let status: tinfl_status = tinfl_decompress(
      &mut decomp,
      in_buf[in_buf_ofs..].as_ptr(),
      &mut in_buf_size,
      dict[].as_ptr(),
      dict[dict_ofs..].as_ptr(),
      &mut dst_buf_size,
      (flags & !(TINFL_FLAG_HAS_MORE_INPUT | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF)));

    // Increase input buffer to reflect data copied out.
    in_buf_ofs += in_buf_size;

    if (dst_buf_size > 0) && ! put_buf_func.call_mut((dict[dict_ofs..dict_ofs+dst_buf_size],)) {break;}
    if status != TINFL_STATUS_HAS_MORE_OUTPUT {
      result = status == TINFL_STATUS_DONE;
      break;
    }
    dict_ofs = (dict_ofs + dst_buf_size) & (TINFL_LZ_DICT_SIZE - 1);
  }
  (result, in_buf_ofs)
}

// ------------------- Low-level Compression (independent from all decompression API's)

// Purposely making these tables static for faster init and thread safety.
const s_tdefl_len_sym: [u16, ..256] = [
  257,258,259,260,261,262,263,264,265,265,266,266,267,267,268,268,269,269,269,269,270,270,270,270,271,271,271,271,272,272,272,272,
  273,273,273,273,273,273,273,273,274,274,274,274,274,274,274,274,275,275,275,275,275,275,275,275,276,276,276,276,276,276,276,276,
  277,277,277,277,277,277,277,277,277,277,277,277,277,277,277,277,278,278,278,278,278,278,278,278,278,278,278,278,278,278,278,278,
  279,279,279,279,279,279,279,279,279,279,279,279,279,279,279,279,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,
  281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,281,
  282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
  283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,283,
  284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,284,285 ];

const s_tdefl_len_extra: [u8, ..256] = [
  0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0 ];

const s_tdefl_small_dist_sym: [u8, ..512] = [
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

const s_tdefl_small_dist_extra: [u8, ..512] = [
  0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7 ];

const s_tdefl_large_dist_sym: [u8, ..128] = [
  0,0,18,19,20,20,21,21,22,22,22,22,23,23,23,23,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,26,26,26,
  26,26,26,26,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,
  28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29 ];

const s_tdefl_large_dist_extra: [u8, ..128] = [
  0,0,8,8,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
  12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
  13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13 ];

#[packed]
struct tdefl_sym_freq { m_key: u16, m_sym_index: u16 }
/// Radix sorts tdefl_sym_freq[] array by 16-bit key m_key. Returns ptr to sorted values.
fn tdefl_radix_sort_syms <'a>(num_syms: uint, pSyms0: &'a mut [tdefl_sym_freq], pSyms1: &'a mut [tdefl_sym_freq]) -> &'a mut [tdefl_sym_freq]
{
  let total_passes: uint = 2; let i: uint = 0; let hist = [0u, ..256 * 2];
  let mut pCur_syms: &mut [tdefl_sym_freq] = pSyms0; let mut pNew_syms: &mut [tdefl_sym_freq] = pSyms1;
  while i < num_syms { let freq: uint = pSyms0[i].m_key as uint; hist[freq & 0xFF]+=1; hist[256 + ((freq >> 8) & 0xFF)]+=1; i+=1}
  while (total_passes > 1) && (num_syms == hist[(total_passes - 1) * 256]) {total_passes-=1};

  let pass_shift: uint = 0; let pass: uint = 0;
  while pass < total_passes
  {
    let pHist: &[uint] = hist[pass << 8 .. 256];
    let mut offsets: [uint, ..256]; let cur_ofs: uint = 0;
    i = 0;
    while i < 256 { offsets[i] = cur_ofs; cur_ofs += pHist[i]; i+=1 }
    i = 0;
    while i < num_syms {
      let temp_index = (pCur_syms[i].m_key >> pass_shift) as uint & 0xFF;
      pNew_syms[offsets[temp_index]] = pCur_syms[i];
      offsets[temp_index] += 1;
      i+=1;
    }
    { let t = pCur_syms; pCur_syms = pNew_syms; pNew_syms = t; }
    pass += 1; pass_shift += 8;
  }
  return pCur_syms;
}

/// tdefl_calculate_minimum_redundancy() originally written by: Alistair Moffat, alistair@cs.mu.oz.au, Jyrki Katajainen, jyrki@diku.dk, November 1996.
fn tdefl_calculate_minimum_redundancy(A: &mut [tdefl_sym_freq], n: uint)
{
  let root: uint = 0; let leaf: uint = 2; let mut next: uint = 1; let avbl: uint = 1; let used: uint = 0; let dpth: c_int = 0;
  if n==0 {return;} else {if n==1 { A[0].m_key = 1; return; }}
  A[0].m_key += A[1].m_key;
  while (next < n-1)
  {
    if (leaf>=n || A[root].m_key<A[leaf].m_key) { A[next].m_key = A[root].m_key; A[root].m_key = next as u16; root+=1; } else {A[next].m_key = A[leaf].m_key; leaf+=1;}
    if (leaf>=n || (root<next && A[root].m_key<A[leaf].m_key)) { A[next].m_key = (A[next].m_key + A[root].m_key) as u16; A[root].m_key = next as u16; root+=1 }
    else {A[next].m_key = (A[next].m_key + A[leaf].m_key) as u16; leaf+=1;}
    next+=1;
  }
  A[n-2].m_key = 0; next=n-3; while next>=0 {A[next].m_key = A[A[next].m_key as uint].m_key+1; next-=1;}
  root = n-2; next = n-1;
  while (avbl>0)
  {
    while (root>=0 && A[root].m_key as c_int == dpth) { used+=1; root-=1; }
    while (avbl>used) { A[next].m_key = dpth as u16; next-=1; avbl-=1; }
    avbl = 2*used; dpth+=1; used = 0;
  }
}

// Limits canonical Huffman code table's max code size.
const TDEFL_MAX_SUPPORTED_HUFF_CODESIZE: uint = 32;
fn tdefl_huffman_enforce_max_code_size(pNum_codes: &mut[uint], code_list_len: uint, max_code_size: uint)
{
  let i: uint; let total: u32 = 0; if (code_list_len <= 1) {return;}
  i = max_code_size + 1;
  while i <= TDEFL_MAX_SUPPORTED_HUFF_CODESIZE {pNum_codes[max_code_size] += pNum_codes[i]; i+=1;}
  i = max_code_size;
  while i > 0 {total += ((pNum_codes[i] as u32) << (max_code_size - i)); i-=1;}
  while total != (1 << max_code_size)
  {
    pNum_codes[max_code_size]-=1;
    i = max_code_size - 1;
    while i > 0 {if pNum_codes[i] != 0 { pNum_codes[i]-=1; pNum_codes[i + 1] += 2; break; }; i-=1;}
    total-=1;
  }
}

fn tdefl_optimize_huffman_table(d: &mut tdefl_compressor, table_num: uint, table_len: uint, code_size_limit: uint, static_table: bool)
{
  let i: uint = 0; let j: uint; let l: uint; let num_codes= [0 as uint, ..1 + TDEFL_MAX_SUPPORTED_HUFF_CODESIZE]; let next_code= [0 as uint, ..TDEFL_MAX_SUPPORTED_HUFF_CODESIZE + 1];
  if static_table
  {
    while i < table_len { num_codes[d.m_huff_code_sizes[table_num][i] as uint]+=1; i+=1 }
  }
  else
  {
    let syms0: [tdefl_sym_freq, ..TDEFL_MAX_HUFF_SYMBOLS]; let syms1: [tdefl_sym_freq, ..TDEFL_MAX_HUFF_SYMBOLS]; let pSyms: &mut[tdefl_sym_freq];
    let num_used_syms: uint = 0;
    let pSym_count: &[u16] = d.m_huff_count[table_num];
    while i < table_len { if pSym_count[i]!=0 { syms0[num_used_syms].m_key = pSym_count[i] as u16; syms0[num_used_syms].m_sym_index = i as u16; num_used_syms+=1}; i+=1 }

    pSyms = tdefl_radix_sort_syms(num_used_syms, syms0, syms1); tdefl_calculate_minimum_redundancy(pSyms, num_used_syms);

    i = 0;
    while i < num_used_syms { num_codes[pSyms[i].m_key as uint]+=1; i+=1}

    tdefl_huffman_enforce_max_code_size(&mut num_codes, num_used_syms, code_size_limit);

    for i in d.m_huff_code_sizes[table_num].iter_mut() {*i = 0u8};
    for i in d.m_huff_codes[table_num].iter_mut() {*i = 0u16};
    i = 1; j = num_used_syms;
    while i <= code_size_limit {
      l = num_codes[i];
      while l > 0 { j-=1; d.m_huff_code_sizes[table_num][pSyms[j].m_sym_index as uint] = i as u8; l-=1 }
      i+=1
    }
  }

  next_code[1] = 0; j = 0; i = 2; while i <= code_size_limit { j = ((j + num_codes[i - 1]) << 1); next_code[i] = j; i+=1 };

  i = 0;
  while i < table_len
  {
    let rev_code: uint = 0; let code: uint; let code_size: uint = d.m_huff_code_sizes[table_num][i] as uint;
    if code_size == 0 {continue;}
    code = next_code[code_size]; next_code[code_size]+=1; l = code_size; while l > 0 { rev_code = (rev_code << 1) | (code & 1); l-=1; code >>= 1};
    d.m_huff_codes[table_num][i] = rev_code as u16;
    i+=1
  }
}

const s_tdefl_packed_code_size_syms_swizzle: [u8, ..19] = [ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ];

fn tdefl_start_dynamic_block(d: &mut tdefl_compressor)
{
  let mut num_lit_codes: uint = 286;
  let mut num_dist_codes: uint = 30;
  let mut num_bit_lengths: uint = 18;
  let mut i: uint;
  let total_code_sizes_to_pack: uint;
  let num_packed_code_sizes: uint = 0;
  let rle_z_count: mz_uint = 0;
  let rle_repeat_count: mz_uint = 0;
  let packed_code_sizes_index: uint = 0;
  let code_sizes_to_pack: [u8, ..TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1];
  let packed_code_sizes: [u8, ..TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1];
  let prev_code_size: u8 = 0xFF;

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: uint = $b;
      let len: uint = $l;
      assert!(bits <= ((1u << len) - 1u));
      d.m_bit_buffer |= (bits << d.m_bits_in); d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf = d.m_pOutput_buf.offset(1);
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  )

  macro_rules! TDEFL_RLE_PREV_CODE_SIZE( () =>
    ({
      if rle_repeat_count != 0 {
        if (rle_repeat_count < 3) {
          d.m_huff_count[2][prev_code_size as uint] = d.m_huff_count[2][prev_code_size as uint] + (rle_repeat_count as u16);
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
  )

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
  )

  d.m_huff_count[0][256] = 1;

  tdefl_optimize_huffman_table(d, 0, TDEFL_MAX_HUFF_SYMBOLS_0, 15, false);
  tdefl_optimize_huffman_table(d, 1, TDEFL_MAX_HUFF_SYMBOLS_1, 15, false);

  while num_lit_codes > 257 { if (d.m_huff_code_sizes[0][num_lit_codes - 1]!=0) {break;}; num_lit_codes-=1; }
  while num_dist_codes > 1 { if (d.m_huff_code_sizes[1][num_dist_codes - 1]!=0) {break;}; num_dist_codes-=1; }

  copy_memory(code_sizes_to_pack.as_mut_ptr(), &d.m_huff_code_sizes[0][0], num_lit_codes);
  copy_memory(code_sizes_to_pack.as_mut_ptr().offset(num_lit_codes as int), &d.m_huff_code_sizes[1][0], num_dist_codes);
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
      if (code_size != prev_code_size)
      {
        TDEFL_RLE_PREV_CODE_SIZE!();
        d.m_huff_count[2][code_size as uint] = (d.m_huff_count[2][code_size as uint] + 1) as u16; packed_code_sizes[num_packed_code_sizes] = code_size; num_packed_code_sizes+=1;
      }
      else {
        rle_repeat_count+=1;
        if (rle_repeat_count == 6)
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

  while num_bit_lengths >= 0 { if d.m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[num_bit_lengths] as uint] != 0 {break;}; num_bit_lengths-=1 }
  num_bit_lengths = max(4, (num_bit_lengths + 1)); TDEFL_PUT_BITS!(num_bit_lengths - 4, 4);
  i = 0;
  while i < num_bit_lengths { TDEFL_PUT_BITS!(d.m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[i] as uint] as uint, 3); i+=1 }

  while packed_code_sizes_index < num_packed_code_sizes
  {
    let code: uint = packed_code_sizes[packed_code_sizes_index] as uint; packed_code_sizes_index+=1; assert!(code < TDEFL_MAX_HUFF_SYMBOLS_2);
    TDEFL_PUT_BITS!(d.m_huff_codes[2][code] as uint, d.m_huff_code_sizes[2][code] as uint);
    if code >= 16 {TDEFL_PUT_BITS!(packed_code_sizes[packed_code_sizes_index] as uint, "\02\03\07"[code - 16]); packed_code_sizes_index+=1;}
  }
}

fn tdefl_start_static_block(d: &mut tdefl_compressor)
{
  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: uint = $b;
      let len: uint = $l;
      assert!(bits <= ((1u << len) - 1u));
      d.m_bit_buffer |= (bits << d.m_bits_in); d.m_bits_in += len;
      while d.m_bits_in >= 8 {
        if d.m_pOutput_buf < d.m_pOutput_buf_end{
          *d.m_pOutput_buf = d.m_bit_buffer as u8;
          d.m_pOutput_buf = d.m_pOutput_buf.offset(1);
        }
        d.m_bit_buffer >>= 8;
        d.m_bits_in -= 8;
      }
    };);
  )

  for i in d.m_huff_code_sizes[0][  0..144].iter_mut() { *i = 8 }
  for i in d.m_huff_code_sizes[0][144..256].iter_mut() { *i = 9 }
  for i in d.m_huff_code_sizes[0][256..280].iter_mut() { *i = 7 }
  for i in d.m_huff_code_sizes[0][280..288].iter_mut() { *i = 8 }

  for i in d.m_huff_code_sizes[1][  0.. 32].iter_mut() { *i = 5 }

  tdefl_optimize_huffman_table(d, 0, 288, 15, true);
  tdefl_optimize_huffman_table(d, 1, 32, 15, true);

  TDEFL_PUT_BITS!(1, 2);
}

const mz_bitmasks: [mz_uint, ..17] = [ 0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF ];

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
fn tdefl_compress_lz_codes(d: &mut tdefl_compressor) -> bool
{
  let flags: mz_uint;
  let pLZ_codes: *const u8 = d.m_lz_code_buf;
  let pOutput_buf: *mut u8 = d.m_pOutput_buf;
  let pLZ_code_buf_end: *const u8 = d.m_pLZ_code_buf;
  let bit_buffer: u64 = d.m_bit_buffer;
  let bits_in: mz_uint = d.m_bits_in;

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: mz_uint = $b;
      let len: mz_uint = $l;
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
  )

  macro_rules! TDEFL_PUT_BITS_FAST( ($b:expr, $l:expr) =>
    ({
      bit_buffer |= (($b as u64) << bits_in);
      bits_in += $l;
    });
  )

  flags = 1;
  while pLZ_codes < pLZ_code_buf_end
  {
    if (flags == 1) {
      flags = *pLZ_codes | 0x100;
      pLZ_codes+=1;
    }

    if (flags & 1)
    {
      let s0: mz_uint;
      let s1: mz_uint;
      let n0: mz_uint;
      let n1: mz_uint;
      let sym: mz_uint;
      let num_extra_bits: mz_uint;
      let match_len: mz_uint = pLZ_codes[0];
      let match_dist: mz_uint = *((pLZ_codes + 1) as *const u16); pLZ_codes += 3;

      assert!(d.m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][s_tdefl_len_sym[match_len]], d.m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS_FAST!(match_len & mz_bitmasks[s_tdefl_len_extra[match_len]], s_tdefl_len_extra[match_len]);

      // This sequence coaxes MSVC into using cmov's vs. jmp's.
      s0 = s_tdefl_small_dist_sym[match_dist & 511];
      n0 = s_tdefl_small_dist_extra[match_dist & 511];
      s1 = s_tdefl_large_dist_sym[match_dist >> 8];
      n1 = s_tdefl_large_dist_extra[match_dist >> 8];
      sym = if match_dist < 512 {s0} else {s1};
      num_extra_bits = if match_dist < 512 {n0} else {n1};

      assert!(d.m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS_FAST!(d.m_huff_codes[1][sym], d.m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS_FAST!(match_dist & mz_bitmasks[num_extra_bits], num_extra_bits);
    }
    else
    {
      let lit: mz_uint = *pLZ_codes;
      pLZ_codes+=1;
      assert!(d.m_huff_code_sizes[0][lit]);
      TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit]);

      if (((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end))
      {
        flags >>= 1;
        lit = *pLZ_codes;
        pLZ_codes+=1;
        assert!(d.m_huff_code_sizes[0][lit]);
        TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit]);

        if (((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end))
        {
          flags >>= 1;
          lit = *pLZ_codes;
          pLZ_codes+=1;
          assert!(d.m_huff_code_sizes[0][lit]);
          TDEFL_PUT_BITS_FAST!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit]);
        }
      }
    }

    if (pOutput_buf >= d.m_pOutput_buf_end){
      return false;
    }

    *(pOutput_buf as *mut u64) = bit_buffer;
    pOutput_buf += (bits_in >> 3);
    bit_buffer >>= (bits_in & !7);
    bits_in &= 7;

    flags >>= 1;
  }

  d.m_pOutput_buf = pOutput_buf;
  d.m_bits_in = 0;
  d.m_bit_buffer = 0;

  while (bits_in)
  {
    let n: u32 = min(bits_in, 16);
    TDEFL_PUT_BITS!((bit_buffer as mz_uint) & mz_bitmasks[n], n);
    bit_buffer >>= n;
    bits_in -= n;
  }

  TDEFL_PUT_BITS!(d.m_huff_codes[0][256], d.m_huff_code_sizes[0][256]);

  return (d.m_pOutput_buf < d.m_pOutput_buf_end);
}
#[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
fn tdefl_compress_lz_codes(d: &mut tdefl_compressor) -> bool
{
  let flags: mz_uint = 1;
  let pLZ_codes: *const u8 = d.m_lz_code_buf;

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: mz_uint = $b;
      let len: mz_uint = $l;
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
  )

  while pLZ_codes < d.m_pLZ_code_buf
  {
    if (flags == 1){
      flags = *pLZ_codes | 0x100;
      pLZ_codes += 1;
    }
    if (flags & 1)
    {
      let sym: mz_uint; let num_extra_bits: mz_uint;
      let match_len: mz_uint = pLZ_codes[0]; let match_dist: mz_uint = (pLZ_codes[1] | (pLZ_codes[2] << 8)); pLZ_codes += 3;

      assert!(d.m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS!(d.m_huff_codes[0][s_tdefl_len_sym[match_len]], d.m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS!(match_len & mz_bitmasks[s_tdefl_len_extra[match_len]], s_tdefl_len_extra[match_len]);

      if (match_dist < 512)
      {
        sym = s_tdefl_small_dist_sym[match_dist]; num_extra_bits = s_tdefl_small_dist_extra[match_dist];
      }
      else
      {
        sym = s_tdefl_large_dist_sym[match_dist >> 8]; num_extra_bits = s_tdefl_large_dist_extra[match_dist >> 8];
      }
      assert!(d.m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS!(d.m_huff_codes[1][sym], d.m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS!(match_dist & mz_bitmasks[num_extra_bits], num_extra_bits);
    }
    else
    {
      let lit: mz_uint = *pLZ_codes; pLZ_codes+=1;
      assert!(d.m_huff_code_sizes[0][lit]);
      TDEFL_PUT_BITS!(d.m_huff_codes[0][lit], d.m_huff_code_sizes[0][lit]);
    }
    flags >>= 1;
  }

  TDEFL_PUT_BITS!(d.m_huff_codes[0][256], d.m_huff_code_sizes[0][256]);

  return (d.m_pOutput_buf < d.m_pOutput_buf_end);
}
// #endif // MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN && MINIZ_HAS_64BIT_REGISTERS

fn tdefl_compress_block(d: &mut tdefl_compressor, static_block: bool) -> bool
{
  if (static_block) {
    tdefl_start_static_block(d);
  } else {
    tdefl_start_dynamic_block(d);
  } return tdefl_compress_lz_codes(d);
}

fn tdefl_flush_block(d: &mut tdefl_compressor, flush: int) -> int
{
  let saved_bit_buf: mz_uint; let saved_bits_in: mz_uint;
  let pSaved_output_buf: *const u8;
  let comp_block_succeeded: bool = false;
  let n: int; let use_raw_block: int = ((d.m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS) != 0) && (d.m_lookahead_pos - d.m_lz_code_buf_dict_pos) <= d.m_dict_size;
  let pOutput_buf_start: *const u8 = if ((d.m_pPut_buf_func.is_null()) && ((*d.m_pOut_buf_size - d.m_out_buf_ofs) >= TDEFL_OUT_BUF_SIZE)) {(d.m_pOut_buf as *const u8) + d.m_out_buf_ofs} else {d.m_output_buf};

  macro_rules! TDEFL_PUT_BITS( ($b:expr, $l:expr) =>
    ({
      let bits: mz_uint = $b;
      let len: mz_uint = $l;
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
  )

  d.m_pOutput_buf = pOutput_buf_start;
  d.m_pOutput_buf_end = d.m_pOutput_buf + TDEFL_OUT_BUF_SIZE - 16;

  assert!(!d.m_output_flush_remaining);
  d.m_output_flush_ofs = 0;
  d.m_output_flush_remaining = 0;

  *d.m_pLZ_flags = (*d.m_pLZ_flags >> d.m_num_flags_left) as u8;
  d.m_pLZ_code_buf -= (d.m_num_flags_left == 8);

  if ((d.m_flags & TDEFL_WRITE_ZLIB_HEADER) && (!d.m_block_index))
  {
    TDEFL_PUT_BITS!(0x78, 8); TDEFL_PUT_BITS!(0x01, 8);
  }

  TDEFL_PUT_BITS!(flush == TDEFL_FINISH, 1);

  pSaved_output_buf = d.m_pOutput_buf; saved_bit_buf = d.m_bit_buffer; saved_bits_in = d.m_bits_in;

  if (!use_raw_block){
    comp_block_succeeded = tdefl_compress_block(d, (d.m_flags & TDEFL_FORCE_ALL_STATIC_BLOCKS) || (d.m_total_lz_bytes < 48));
  }

  // If the block gets expanded, forget the current contents of the output buffer and send a raw block instead.
  if ( ((use_raw_block) || ((d.m_total_lz_bytes) && ((d.m_pOutput_buf - pSaved_output_buf + 1u) >= d.m_total_lz_bytes))) &&
       ((d.m_lookahead_pos - d.m_lz_code_buf_dict_pos) <= d.m_dict_size) )
  {
    let i: mz_uint; d.m_pOutput_buf = pSaved_output_buf; d.m_bit_buffer = saved_bit_buf; d.m_bits_in = saved_bits_in;
    TDEFL_PUT_BITS!(0, 2);
    if (d.m_bits_in) { TDEFL_PUT_BITS!(0, 8 - d.m_bits_in); }
    i = 2;
    while i > 0
    {
      TDEFL_PUT_BITS!(d.m_total_lz_bytes & 0xFFFF, 16);
      i -= 1; d.m_total_lz_bytes ^= 0xFFFF;
    }
    i = 0;
    while i < d.m_total_lz_bytes
    {
      TDEFL_PUT_BITS!(d.m_dict[(d.m_lz_code_buf_dict_pos + i) & TDEFL_LZ_DICT_SIZE_MASK], 8);
      i += 1;
    }
  }
  // Check for the extremely unlikely (if not impossible) case of the compressed block not fitting into the output buffer when using dynamic codes.
  else if (!comp_block_succeeded)
  {
    d.m_pOutput_buf = pSaved_output_buf; d.m_bit_buffer = saved_bit_buf; d.m_bits_in = saved_bits_in;
    tdefl_compress_block(d, true);
  }

  if (flush)
  {
    if (flush == TDEFL_FINISH)
    {
      if (d.m_bits_in) { TDEFL_PUT_BITS!(0, 8 - d.m_bits_in); }
      if (d.m_flags & TDEFL_WRITE_ZLIB_HEADER) {
        let i: mz_uint; let a: mz_uint = d.m_adler32; i = 0; while i < 4 { TDEFL_PUT_BITS!((a >> 24) & 0xFF, 8); a <<= 8; i+=1; }
      }
    }
    else
    {
      let i: mz_uint; let z: mz_uint = 0; TDEFL_PUT_BITS!(0, 3); if (d.m_bits_in) { TDEFL_PUT_BITS!(0, 8 - d.m_bits_in); } i = 2; while i > 0 { TDEFL_PUT_BITS!(z & 0xFFFF, 16); i -= 1; z ^= 0xFFFF; }
    }
  }

  assert!(d.m_pOutput_buf < d.m_pOutput_buf_end);

  set_memory(&d.m_huff_count[0][0], 0, size_of(d.m_huff_count[0][0]) * TDEFL_MAX_HUFF_SYMBOLS_0);
  set_memory(&d.m_huff_count[1][0], 0, size_of(d.m_huff_count[1][0]) * TDEFL_MAX_HUFF_SYMBOLS_1);

  d.m_pLZ_code_buf = d.m_lz_code_buf + 1; d.m_pLZ_flags = d.m_lz_code_buf; d.m_num_flags_left = 8; d.m_lz_code_buf_dict_pos += d.m_total_lz_bytes; d.m_total_lz_bytes = 0; d.m_block_index+=1;

  if ((n = (d.m_pOutput_buf - pOutput_buf_start) as c_int) != 0)
  {
    if (d.m_pPut_buf_func)
    {
      *d.m_pIn_buf_size = d.m_pSrc - d.m_pIn_buf as *const u8;
      if (! d.m_pPut_buf_func.call_mut((d.m_output_buf, n)) ) {
        return (d.m_prev_return_status = TDEFL_STATUS_PUT_BUF_FAILED);
      }
    }
    else if (pOutput_buf_start == d.m_output_buf)
    {
      let bytes_to_copy: int = min(n as size_t, (*d.m_pOut_buf_size - d.m_out_buf_ofs) as size_t) as c_int;
      copy_memory((d.m_pOut_buf as *mut u8) + d.m_out_buf_ofs, d.m_output_buf, bytes_to_copy);
      d.m_out_buf_ofs += bytes_to_copy;
      if ((n -= bytes_to_copy) != 0)
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

  return d.m_output_flush_remaining;
}

// #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES
macro_rules! TDEFL_READ_UNALIGNED_WORD(($p:expr) => (*($p as *const u16)); )
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn tdefl_find_match(d: &mut tdefl_compressor, lookahead_pos: mz_uint, max_dist: mz_uint, max_match_len: mz_uint, pMatch_dist: *mut mz_uint, pMatch_len: *mut mz_uint)
{
  let dist: mz_uint;
  let pos: mz_uint = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
  let match_len: mz_uint = *pMatch_len;
  let probe_pos: mz_uint = pos;
  let next_probe_pos: mz_uint;
  let probe_len: mz_uint;
  let num_probes_left: mz_uint = d.m_max_probes[match_len >= 32];
  let s: *const u16 = (d.m_dict + pos) as *const u16;
  let p: *const u16;
  let q: *const u16;
  let c01: u16 = TDEFL_READ_UNALIGNED_WORD!(&d.m_dict[pos + match_len - 1]);
  let s01: u16 = TDEFL_READ_UNALIGNED_WORD!(s);
  assert!(max_match_len <= TDEFL_MAX_MATCH_LEN); if max_match_len <= match_len {return;}
  loop {
    loop {
      num_probes_left -= 1;
      if num_probes_left == 0 {return;}
      macro_rules! TDEFL_PROBE( () => ({
        next_probe_pos = d.m_next[probe_pos];
        if (!next_probe_pos) || ((dist = (lookahead_pos - next_probe_pos) as u16) > max_dist) {return;}
        probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;
        if TDEFL_READ_UNALIGNED_WORD!(&d.m_dict[probe_pos + match_len - 1]) == c01 {break;}
      });)
      TDEFL_PROBE!(); TDEFL_PROBE!(); TDEFL_PROBE!();
    }
    if !dist {break;}; q = (d.m_dict + probe_pos) as *const u16; if TDEFL_READ_UNALIGNED_WORD!(q) != s01 {continue;}; p = s; probe_len = 32;
    loop {
      if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
        if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
          if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
            if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
              if {probe_len -= 1; probe_len > 0} {
                continue;
              }
            }
          }
        }
      }
      break;
    }
    if !probe_len
    {
      *pMatch_dist = dist; *pMatch_len = min(max_match_len, TDEFL_MAX_MATCH_LEN); break;
    }
    else if (probe_len = (((p - s) as mz_uint) * 2) + (*(p as *const u8) == *(q as *const u8)) as mz_uint) > match_len
    {
      *pMatch_dist = dist; if (*pMatch_len = match_len = min(max_match_len, probe_len)) == max_match_len {break;}
      c01 = TDEFL_READ_UNALIGNED_WORD!(&d.m_dict[pos + match_len - 1]);
    }
  }
}
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn tdefl_find_match(d: &mut tdefl_compressor, lookahead_pos: mz_uint, max_dist: mz_uint, max_match_len: mz_uint, pMatch_dist: *mut mz_uint, pMatch_len: *mut mz_uint)
{
  let dist: mz_uint;
  let pos: mz_uint = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
  let match_len: mz_uint = *pMatch_len;
  let probe_pos: mz_uint = pos;
  let next_probe_pos: mz_uint;
  let probe_len: mz_uint;
  let num_probes_left: mz_uint = d.m_max_probes[match_len >= 32];
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
      );)
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
fn tdefl_compress_fast(d: &mut tdefl_compressor) -> bool
{
  // Faster, minimally featured LZRW1-style match+parse loop with better register utilization. Intended for applications where raw throughput is valued more highly than ratio.
  let lookahead_pos: uint = d.m_lookahead_pos;
  let lookahead_size: uint = d.m_lookahead_size;
  let dict_size: uint = d.m_dict_size;
  let total_lz_bytes: uint = d.m_total_lz_bytes;
  let num_flags_left: mz_uint = d.m_num_flags_left;
  let pLZ_code_buf: *mut u8 = d.m_pLZ_code_buf; let pLZ_flags: *mut u8 = d.m_pLZ_flags;
  let cur_pos: uint = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;

  while ((d.m_src_buf_left) || ((d.m_flush) && (lookahead_size)))
  {
    let TDEFL_COMP_FAST_LOOKAHEAD_SIZE: mz_uint = 4096;
    let dst_pos: mz_uint = (lookahead_pos + lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
    let num_bytes_to_process: mz_uint = min(d.m_src_buf_left, TDEFL_COMP_FAST_LOOKAHEAD_SIZE - lookahead_size) as mz_uint;
    d.m_src_buf_left -= num_bytes_to_process;
    lookahead_size += num_bytes_to_process;

    while (num_bytes_to_process)
    {
      let n: u32 = min(TDEFL_LZ_DICT_SIZE - dst_pos, num_bytes_to_process);
      copy_memory(d.m_dict + dst_pos, d.m_pSrc, n);
      if (dst_pos < (TDEFL_MAX_MATCH_LEN - 1)){
        copy_memory(d.m_dict + TDEFL_LZ_DICT_SIZE + dst_pos, d.m_pSrc, min(n, (TDEFL_MAX_MATCH_LEN - 1) - dst_pos));
      }
      d.m_pSrc += n;
      dst_pos = (dst_pos + n) & TDEFL_LZ_DICT_SIZE_MASK;
      num_bytes_to_process -= n;
    }

    dict_size = min(TDEFL_LZ_DICT_SIZE - lookahead_size, dict_size);
    if (!d.m_flush) && (lookahead_size < TDEFL_COMP_FAST_LOOKAHEAD_SIZE) {break;}

    while (lookahead_size >= 4)
    {
      let cur_match_dist: mz_uint;
      let cur_match_len: mz_uint = 1;
      let pCur_dict: *const u8 = d.m_dict + cur_pos;
      let first_trigram: mz_uint = (*(pCur_dict as *const u32)) & 0xFFFFFF;
      let hash: mz_uint = (first_trigram ^ (first_trigram >> (24 - (TDEFL_LZ_HASH_BITS - 8)))) & TDEFL_LEVEL1_HASH_SIZE_MASK;
      let probe_pos: mz_uint = d.m_hash[hash];
      d.m_hash[hash] = lookahead_pos as u16;

      if (((cur_match_dist = (lookahead_pos - probe_pos) as u16) <= dict_size) && ((*((d.m_dict + (probe_pos &= TDEFL_LZ_DICT_SIZE_MASK)) as *const u32) & 0xFFFFFF) == first_trigram))
      {
        let p: *const u16 = pCur_dict as *const u16;
        let q: *const u16 = (d.m_dict + probe_pos) as *const u16;
        let probe_len: u32 = 32;
        loop {
          if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
            if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
              if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
                if {p += 1; q += 1; (TDEFL_READ_UNALIGNED_WORD!(p)) == (TDEFL_READ_UNALIGNED_WORD!(q))} {
                  if {probe_len -= 1; probe_len > 0} {
                    continue;
                  }
                }
              }
            }
          }
          break;
        }
        cur_match_len = (((p - pCur_dict as *const u16) as mz_uint) * 2) + (*(p as *const u8) == *(q as *const u8)) as mz_uint;
        if !probe_len {
          cur_match_len = if cur_match_dist {TDEFL_MAX_MATCH_LEN} else {0};
        }

        if ((cur_match_len < TDEFL_MIN_MATCH_LEN) || ((cur_match_len == TDEFL_MIN_MATCH_LEN) && (cur_match_dist >= 8u*1024u)))
        {
          cur_match_len = 1;
          *pLZ_code_buf = first_trigram as u8; pLZ_code_buf+=1;
          *pLZ_flags = (*pLZ_flags >> 1) as u8;
          d.m_huff_count[0][first_trigram as u8]+=1;
        }
        else
        {
          let s0: u32; let s1: u32;
          cur_match_len = min(cur_match_len, lookahead_size);

          assert!((cur_match_len >= TDEFL_MIN_MATCH_LEN) && (cur_match_dist >= 1) && (cur_match_dist <= TDEFL_LZ_DICT_SIZE));

          cur_match_dist-=1;

          pLZ_code_buf[0] = (cur_match_len - TDEFL_MIN_MATCH_LEN) as u8;
          *((&pLZ_code_buf[1]) as *mut u16) = cur_match_dist as u16;
          pLZ_code_buf += 3;
          *pLZ_flags = ((*pLZ_flags >> 1) | 0x80) as u8;

          s0 = s_tdefl_small_dist_sym[cur_match_dist & 511];
          s1 = s_tdefl_large_dist_sym[cur_match_dist >> 8];
          d.m_huff_count[1][if cur_match_dist < 512 {s0} else {s1}]+=1;

          d.m_huff_count[0][s_tdefl_len_sym[cur_match_len - TDEFL_MIN_MATCH_LEN]]+=1;
        }
      }
      else
      {
        *pLZ_code_buf = first_trigram as u8; pLZ_code_buf += 1;
        *pLZ_flags = (*pLZ_flags >> 1) as u8;
        d.m_huff_count[0][first_trigram as u8]+=1;
      }

      num_flags_left -= 1;
      if num_flags_left == 0 { num_flags_left = 8; pLZ_flags = pLZ_code_buf+=1; }

      total_lz_bytes += cur_match_len;
      lookahead_pos += cur_match_len;
      dict_size = min(dict_size + cur_match_len, TDEFL_LZ_DICT_SIZE);
      cur_pos = (cur_pos + cur_match_len) & TDEFL_LZ_DICT_SIZE_MASK;
      assert!(lookahead_size >= cur_match_len);
      lookahead_size -= cur_match_len;

      if (pLZ_code_buf > &d.m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE - 8])
      {
        let n: int;
        d.m_lookahead_pos = lookahead_pos; d.m_lookahead_size = lookahead_size; d.m_dict_size = dict_size;
        d.m_total_lz_bytes = total_lz_bytes; d.m_pLZ_code_buf = pLZ_code_buf; d.m_pLZ_flags = pLZ_flags; d.m_num_flags_left = num_flags_left;
        if ((n = tdefl_flush_block(d, 0)) != 0){
          return if n < 0 {false} else {true};
        }
        total_lz_bytes = d.m_total_lz_bytes; pLZ_code_buf = d.m_pLZ_code_buf; pLZ_flags = d.m_pLZ_flags; num_flags_left = d.m_num_flags_left;
      }
    }

    while (lookahead_size)
    {
      let lit: u8 = d.m_dict[cur_pos];

      total_lz_bytes+=1;
      *pLZ_code_buf = lit; pLZ_code_buf += 1;
      *pLZ_flags = (*pLZ_flags >> 1) as u8;
      num_flags_left -= 1;
      if num_flags_left == 0 { num_flags_left = 8; pLZ_flags = pLZ_code_buf; pLZ_code_buf+=1; }

      d.m_huff_count[0][lit]+=1;

      lookahead_pos+=1;
      dict_size = min(dict_size + 1, TDEFL_LZ_DICT_SIZE);
      cur_pos = (cur_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK;
      lookahead_size-=1;

      if (pLZ_code_buf > &d.m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE - 8])
      {
        let n: int;
        d.m_lookahead_pos = lookahead_pos; d.m_lookahead_size = lookahead_size; d.m_dict_size = dict_size;
        d.m_total_lz_bytes = total_lz_bytes; d.m_pLZ_code_buf = pLZ_code_buf; d.m_pLZ_flags = pLZ_flags; d.m_num_flags_left = num_flags_left;
        if ((n = tdefl_flush_block(d, 0)) != 0){
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
fn tdefl_record_literal(d: &mut tdefl_compressor, lit: u8)
{
  d.m_total_lz_bytes+=1;
  *d.m_pLZ_code_buf = lit; d.m_pLZ_code_buf += 1;
  *d.m_pLZ_flags = (*d.m_pLZ_flags >> 1) as u8; d.m_num_flags_left -= 1; if d.m_num_flags_left == 0 { d.m_num_flags_left = 8; d.m_pLZ_flags = d.m_pLZ_code_buf; d.m_pLZ_code_buf += 1; }
  d.m_huff_count[0][lit] += 1;
}

#[inline(always)]
fn tdefl_record_match(d: &mut tdefl_compressor, match_len: mz_uint, match_dist: mz_uint)
{
  let s0: u32;
  let s1: u32;

  assert!((match_len >= TDEFL_MIN_MATCH_LEN) && (match_dist >= 1) && (match_dist <= TDEFL_LZ_DICT_SIZE));

  d.m_total_lz_bytes += match_len;

  d.m_pLZ_code_buf[0] = (match_len - TDEFL_MIN_MATCH_LEN) as u8;

  match_dist -= 1;
  d.m_pLZ_code_buf[1] = (match_dist & 0xFF) as u8;
  d.m_pLZ_code_buf[2] = (match_dist >> 8) as u8; d.m_pLZ_code_buf += 3;

  *d.m_pLZ_flags = ((*d.m_pLZ_flags >> 1) | 0x80) as u8; d.m_num_flags_left -= 1; if d.m_num_flags_left == 0 { d.m_num_flags_left = 8; d.m_pLZ_flags = d.m_pLZ_code_buf; d.m_pLZ_code_buf += 1; }

  s0 = s_tdefl_small_dist_sym[match_dist & 511]; s1 = s_tdefl_large_dist_sym[(match_dist >> 8) & 127];
  d.m_huff_count[1][if match_dist < 512 {s0} else {s1}] += 1;

  if match_len >= TDEFL_MIN_MATCH_LEN {d.m_huff_count[0][s_tdefl_len_sym[match_len - TDEFL_MIN_MATCH_LEN]] += 1;}
}

fn tdefl_compress_normal(d: &mut tdefl_compressor) -> bool
{
  let pSrc: *const u8 = d.m_pSrc; let src_buf_left: size_t = d.m_src_buf_left;
  let flush: tdefl_flush = d.m_flush;

  while ((src_buf_left) || ((flush) && (d.m_lookahead_size)))
  {
    let len_to_move: mz_uint;
    let cur_match_dist: mz_uint;
    let cur_match_len: mz_uint;
    let cur_pos: mz_uint;
    // Update dictionary and hash chains. Keeps the lookahead size equal to TDEFL_MAX_MATCH_LEN.
    if ((d.m_lookahead_size + d.m_dict_size) >= (TDEFL_MIN_MATCH_LEN - 1))
    {
      let dst_pos: uint = (d.m_lookahead_pos + d.m_lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
      let ins_pos: uint = d.m_lookahead_pos + d.m_lookahead_size - 2;
      let hash: mz_uint = (d.m_dict[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] << TDEFL_LZ_HASH_SHIFT) ^ d.m_dict[(ins_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK];
      let num_bytes_to_process: mz_uint = min(src_buf_left, TDEFL_MAX_MATCH_LEN - d.m_lookahead_size) as mz_uint;
      let pSrc_end: *const u8 = pSrc + num_bytes_to_process;
      src_buf_left -= num_bytes_to_process;
      d.m_lookahead_size += num_bytes_to_process;
      while pSrc != pSrc_end
      {
        let c: u8 = *pSrc; pSrc+=1; d.m_dict[dst_pos] = c; if (dst_pos < (TDEFL_MAX_MATCH_LEN - 1)) {d.m_dict[TDEFL_LZ_DICT_SIZE + dst_pos] = c;}
        hash = ((hash << TDEFL_LZ_HASH_SHIFT) ^ c) & (TDEFL_LZ_HASH_SIZE - 1);
        d.m_next[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] = d.m_hash[hash]; d.m_hash[hash] = ins_pos as u16;
        dst_pos = (dst_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK; ins_pos += 1;
      }
    }
    else
    {
      while ((src_buf_left) && (d.m_lookahead_size < TDEFL_MAX_MATCH_LEN))
      {
        let c: u8 = *pSrc; pSrc += 1;
        let dst_pos: uint = (d.m_lookahead_pos + d.m_lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
        src_buf_left -= 1;
        d.m_dict[dst_pos] = c;
        if (dst_pos < (TDEFL_MAX_MATCH_LEN - 1)){
          d.m_dict[TDEFL_LZ_DICT_SIZE + dst_pos] = c;
        }
        d.m_lookahead_size += 1;
        if (d.m_lookahead_size + d.m_dict_size) >= TDEFL_MIN_MATCH_LEN
        {
          let ins_pos: uint = d.m_lookahead_pos + (d.m_lookahead_size - 1) - 2;
          let hash: mz_uint = ((d.m_dict[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] << (TDEFL_LZ_HASH_SHIFT * 2)) ^ (d.m_dict[(ins_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK] << TDEFL_LZ_HASH_SHIFT) ^ c) & (TDEFL_LZ_HASH_SIZE - 1);
          d.m_next[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] = d.m_hash[hash]; d.m_hash[hash] = ins_pos as u16;
        }
      }
    }
    d.m_dict_size = min(TDEFL_LZ_DICT_SIZE - d.m_lookahead_size, d.m_dict_size);
    if ((!flush) && (d.m_lookahead_size < TDEFL_MAX_MATCH_LEN)){
      break;
    }

    // Simple lazy/greedy parsing state machine.
    len_to_move = 1; cur_match_dist = 0; cur_match_len = if d.m_saved_match_len {d.m_saved_match_len} else {TDEFL_MIN_MATCH_LEN - 1}; cur_pos = d.m_lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
    if (d.m_flags & (TDEFL_RLE_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS))
    {
      if ((d.m_dict_size) && (!(d.m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS)))
      {
        let c: u8 = d.m_dict[(cur_pos - 1) & TDEFL_LZ_DICT_SIZE_MASK];
        cur_match_len = 0; while cur_match_len < d.m_lookahead_size { if d.m_dict[cur_pos + cur_match_len] != c {break;} cur_match_len += 1; }
        if cur_match_len < TDEFL_MIN_MATCH_LEN {cur_match_len = 0;} else {cur_match_dist = 1;}
      }
    }
    else
    {
      tdefl_find_match(d, d.m_lookahead_pos, d.m_dict_size, d.m_lookahead_size, &cur_match_dist, &cur_match_len);
    }
    if (((cur_match_len == TDEFL_MIN_MATCH_LEN) && (cur_match_dist >= 8u*1024u)) || (cur_pos == cur_match_dist) || ((d.m_flags & TDEFL_FILTER_MATCHES) && (cur_match_len <= 5)))
    {
      cur_match_dist = cur_match_len = 0;
    }
    if (d.m_saved_match_len)
    {
      if (cur_match_len > d.m_saved_match_len)
      {
        tdefl_record_literal(d, d.m_saved_lit as u8);
        if (cur_match_len >= 128)
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
        tdefl_record_match(d, d.m_saved_match_len, d.m_saved_match_dist);
        len_to_move = d.m_saved_match_len - 1; d.m_saved_match_len = 0;
      }
    }
    else if (!cur_match_dist){
      tdefl_record_literal(d, d.m_dict[min(cur_pos, size_of(d.m_dict) - 1)]);
    }
    else if ((d.m_greedy_parsing) || (d.m_flags & TDEFL_RLE_MATCHES) || (cur_match_len >= 128))
    {
      tdefl_record_match(d, cur_match_len, cur_match_dist);
      len_to_move = cur_match_len;
    }
    else
    {
      d.m_saved_lit = d.m_dict[min(cur_pos, size_of(d.m_dict) - 1)]; d.m_saved_match_dist = cur_match_dist; d.m_saved_match_len = cur_match_len;
    }
    // Move the lookahead forward by len_to_move bytes.
    d.m_lookahead_pos += len_to_move;
    assert!(d.m_lookahead_size >= len_to_move);
    d.m_lookahead_size -= len_to_move;
    d.m_dict_size = min(d.m_dict_size + len_to_move, TDEFL_LZ_DICT_SIZE);
    // Check if it's time to flush the current LZ codes to the internal output buffer.
    if ( (d.m_pLZ_code_buf > &d.m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE - 8]) ||
         ( (d.m_total_lz_bytes > 31*1024) && ((((((d.m_pLZ_code_buf - d.m_lz_code_buf) as mz_uint) * 115) >> 7) >= d.m_total_lz_bytes) || (d.m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS))) )
    {
      let n: int;
      d.m_pSrc = pSrc; d.m_src_buf_left = src_buf_left;
      if ((n = tdefl_flush_block(d, 0)) != 0){
        return if n < 0 {false} else {true};
      }
    }
  }

  d.m_pSrc = pSrc; d.m_src_buf_left = src_buf_left;
  return true;
}

fn tdefl_flush_output_buffer(d: &mut tdefl_compressor) -> tdefl_status
{
  if (d.m_pIn_buf_size)
  {
    *d.m_pIn_buf_size = d.m_pSrc - d.m_pIn_buf as *const u8;
  }

  if (d.m_pOut_buf_size)
  {
    let n: size_t = min(*d.m_pOut_buf_size - d.m_out_buf_ofs, d.m_output_flush_remaining);
    copy_memory((d.m_pOut_buf as *mut u8) + d.m_out_buf_ofs, d.m_output_buf + d.m_output_flush_ofs, n);
    d.m_output_flush_ofs += n as mz_uint;
    d.m_output_flush_remaining -= n as mz_uint;
    d.m_out_buf_ofs += n;

    *d.m_pOut_buf_size = d.m_out_buf_ofs;
  }

  return if d.m_finished && !d.m_output_flush_remaining {TDEFL_STATUS_DONE} else {TDEFL_STATUS_OKAY};
}

// Compresses a block of data, consuming as much of the specified input buffer as possible, and writing as much compressed data to the specified output buffer as possible.
fn tdefl_compress(d: &mut tdefl_compressor, pIn_buf: *const u8, pIn_buf_size: *const size_t, pOut_buf: *mut u8, pOut_buf_size: *const size_t, flush: tdefl_flush) -> tdefl_status
{
  if (!d)
  {
    if pIn_buf_size {*pIn_buf_size = 0;}
    if pOut_buf_size {*pOut_buf_size = 0;}
    return TDEFL_STATUS_BAD_PARAM;
  }

  d.m_pIn_buf = pIn_buf; d.m_pIn_buf_size = pIn_buf_size;
  d.m_pOut_buf = pOut_buf; d.m_pOut_buf_size = pOut_buf_size;
  d.m_pSrc = pIn_buf as *const u8; if d.m_src_buf_left = pIn_buf_size {*pIn_buf_size} else {0};
  d.m_out_buf_ofs = 0;
  d.m_flush = flush;

  if ( ((d.m_pPut_buf_func != null()) == ((pOut_buf != null()) || (pOut_buf_size != null()))) || (d.m_prev_return_status != TDEFL_STATUS_OKAY) ||
        (d.m_wants_to_finish && (flush != TDEFL_FINISH)) || (pIn_buf_size && *pIn_buf_size && !pIn_buf) || (pOut_buf_size && *pOut_buf_size && !pOut_buf) )
  {
    if pIn_buf_size {*pIn_buf_size = 0;}
    if pOut_buf_size {*pOut_buf_size = 0;}
    return (d.m_prev_return_status = TDEFL_STATUS_BAD_PARAM);
  }
  d.m_wants_to_finish |= (flush == TDEFL_FINISH);

  if ((d.m_output_flush_remaining) || (d.m_finished)){
    return (d.m_prev_return_status = tdefl_flush_output_buffer(d));
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
  fn todo_name_me4(&d: tdefl_compressor) -> bool {
    (((d.m_flags & TDEFL_MAX_PROBES_MASK) == 1) &&
     ((d.m_flags & TDEFL_GREEDY_PARSING_FLAG) != 0) &&
     ((d.m_flags & (TDEFL_FILTER_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS | TDEFL_RLE_MATCHES)) == 0))
  }
  #[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little")))]
  fn todo_name_me4(&d: tdefl_compressor) -> bool { false }

// #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
  if todo_name_me4(d)
  {
    if (!tdefl_compress_fast(d)){
      return d.m_prev_return_status;
    }
  }
  else
// #endif // #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
  {
    if (!tdefl_compress_normal(d)){
      return d.m_prev_return_status;
    }
  }

  if ((d.m_flags & (TDEFL_WRITE_ZLIB_HEADER | TDEFL_COMPUTE_ADLER32)) && (pIn_buf)){
    d.m_adler32 = mz_adler32(d.m_adler32, pIn_buf as *const u8, d.m_pSrc - pIn_buf as *const u8) as u32;
  }

  if ((flush) && (!d.m_lookahead_size) && (!d.m_src_buf_left) && (!d.m_output_flush_remaining))
  {
    if (tdefl_flush_block(d, flush) < 0){
      return d.m_prev_return_status;
    }
    d.m_finished = (flush == TDEFL_FINISH);
    if flush == TDEFL_FULL_FLUSH { for i in d.m_hash.iter_mut() {i = 0u16;}; for i in d.m_next.iter_mut() {i = 0u16;}; d.m_dict_size = 0; }
  }

  return (d.m_prev_return_status = tdefl_flush_output_buffer(d));
}

// tdefl_compress_buffer() is only usable when the tdefl_init() is called with a non-NULL tdefl_put_buf_func_ptr.
// tdefl_compress_buffer() always consumes the entire input buffer.
fn tdefl_compress_buffer(d: &mut tdefl_compressor, pIn_buf: *const c_void, in_buf_size: size_t, flush: tdefl_flush) -> tdefl_status
{
  assert!(d.m_pPut_buf_func); return tdefl_compress(d, pIn_buf, &in_buf_size, null(), null(), flush);
}

// Initializes the compressor.
// There is no corresponding deinit() function because the tdefl API's do not dynamically allocate memory.
// pBut_buf_func: If NULL, output data will be supplied to the specified callback. In this case, the user should call the tdefl_compress_buffer() API for compression.
// If pBut_buf_func is NULL the user should always call the tdefl_compress() API.
// flags: See the above enums (TDEFL_HUFFMAN_ONLY, TDEFL_WRITE_ZLIB_HEADER, etc.)
fn tdefl_init(d: &mut tdefl_compressor, pPut_buf_func: tdefl_put_buf_func_ptr, flags: int) -> tdefl_status
{
  d.m_pPut_buf_func = pPut_buf_func; d.m_pPut_buf_user = null();
  d.m_flags = flags as mz_uint; d.m_max_probes[0] = 1 + ((flags & 0xFFF) + 2) / 3; d.m_greedy_parsing = (flags & TDEFL_GREEDY_PARSING_FLAG) != 0;
  d.m_max_probes[1] = 1 + (((flags & 0xFFF) >> 2) + 2) / 3;
  if (!(flags & TDEFL_NONDETERMINISTIC_PARSING_FLAG)) {for i in d.m_hash.iter_mut() {i = 0};}
  d.m_lookahead_pos = d.m_lookahead_size = d.m_dict_size = d.m_total_lz_bytes = d.m_lz_code_buf_dict_pos = d.m_bits_in = 0;
  d.m_output_flush_ofs = d.m_output_flush_remaining = d.m_finished = d.m_block_index = d.m_bit_buffer = d.m_wants_to_finish = 0;
  d.m_pLZ_code_buf = d.m_lz_code_buf + 1; d.m_pLZ_flags = d.m_lz_code_buf; d.m_num_flags_left = 8;
  d.m_pOutput_buf = d.m_output_buf; d.m_pOutput_buf_end = d.m_output_buf; d.m_prev_return_status = TDEFL_STATUS_OKAY;
  d.m_saved_match_dist = d.m_saved_match_len = d.m_saved_lit = 0; d.m_adler32 = 1;
  d.m_pIn_buf = null(); d.m_pOut_buf = null();
  d.m_pIn_buf_size = null(); d.m_pOut_buf_size = null();
  d.m_flush = TDEFL_NO_FLUSH; d.m_pSrc = null(); d.m_src_buf_left = 0; d.m_out_buf_ofs = 0;
  set_memory(&d.m_huff_count[0][0], 0, size_of(d.m_huff_count[0][0]) * TDEFL_MAX_HUFF_SYMBOLS_0);
  set_memory(&d.m_huff_count[1][0], 0, size_of(d.m_huff_count[1][0]) * TDEFL_MAX_HUFF_SYMBOLS_1);
  return TDEFL_STATUS_OKAY;
}

fn tdefl_get_prev_return_status(d: &mut tdefl_compressor) -> tdefl_status
{
  return d.m_prev_return_status;
}

fn tdefl_get_adler32(d: &mut tdefl_compressor) -> u32
{
  return d.m_adler32;
}

// tdefl_compress_mem_to_output() compresses a block to an output stream. The above helpers use this function internally.
fn tdefl_compress_mem_to_output(in_buf: &[u8], put_buf_func: tdefl_put_buf_func_ptr, flags: int) -> bool
{
  let comp: tdefl_compressor;
  let mut succeeded: bool;
  succeeded = (tdefl_init(&mut comp, put_buf_func, flags) == TDEFL_STATUS_OKAY);
  succeeded = succeeded && (tdefl_compress_buffer(&mut comp, in_buf.as_ptr(), in_buf.len(), TDEFL_FINISH) == TDEFL_STATUS_DONE);
  return succeeded;
}

struct tdefl_output_buffer
{
  m_size: uint, m_capacity: uint,
  m_pBuf: *mut u8,
  m_expandable: bool
}

impl tdefl_output_buffer {
  fn new() -> tdefl_output_buffer {
    tdefl_output_buffer {
      m_size: 0, m_capacity: 0,
      m_pBuf: null() as *mut u8,
      m_expandable: false
    }
  }
}

fn tdefl_output_buffer_putter(pBuf: *const u8, len: uint, pUser: &mut tdefl_output_buffer) -> bool
{
  let p: &mut tdefl_output_buffer = pUser as &mut tdefl_output_buffer;
  let new_size: uint = p.m_size + len;
  if new_size > p.m_capacity
  {
    if !p.m_expandable {return false;};
    let new_capacity: uint = p.m_capacity; let pNew_buf: *mut u8;
    loop { new_capacity = max(128u, new_capacity << 1u); if new_size <= new_capacity {break;} }
    pNew_buf = /*MZ_REALLOC*/(p.m_pBuf, new_capacity) as *mut u8; if !pNew_buf {return false;}
    p.m_pBuf = pNew_buf; p.m_capacity = new_capacity;
  }
  copy_memory((p.m_pBuf as *mut u8) + p.m_size, pBuf, len); p.m_size = new_size;
  return true;
}

// High level compression functions:
// tdefl_compress_mem_to_heap() compresses a block in memory to a heap block allocated via malloc().
// On entry:
//  pSrc_buf, src_buf_len: Pointer and size of source block to compress.
//  flags: The max match finder probes (default is 128) logically OR'd against the above flags. Higher probes are slower but improve compression.
// On return:
//  Function returns a pointer to the compressed data, or NULL on failure.
//  *pOut_len will be set to the compressed data's size, which could be larger than src_buf_len on uncompressible data.
//  The caller must free() the returned block when it's no longer needed.
fn tdefl_compress_mem_to_heap(src_buf: &[u8], pOut_len: *mut uint, flags: int) -> *mut u8
{
  let out_buf = tdefl_output_buffer::new();
  if !pOut_len {return false;} else {*pOut_len = 0;};
  out_buf.m_expandable = true;
  let mut callback = |&mut: pBuf: *const u8, len: uint| {tdefl_output_buffer_putter(pBuf, len, &mut out_buf)};;
  if !tdefl_compress_mem_to_output(src_buf, &mut callback, flags) {return null();};
  *pOut_len = out_buf.m_size; return out_buf.m_pBuf;
}

// tdefl_compress_mem_to_mem() compresses a block in memory to another block in memory.
// Returns 0 on failure.
fn tdefl_compress_mem_to_mem(dst_buf: &mut[u8], src_buf: &[u8], flags: int) -> uint
{
  let out_buf = tdefl_output_buffer::new();
  out_buf.m_pBuf = dst_buf.as_mut_ptr(); out_buf.m_capacity = dst_buf.len();
  let mut callback = |&mut: pBuf: *const u8, len: uint| {tdefl_output_buffer_putter(pBuf, len, &mut out_buf)};;
  if !tdefl_compress_mem_to_output(src_buf, &mut callback, flags) {return 0;};
  return out_buf.m_size;
}

/*
  This is free and unencumbered software released into the public domain.

  Anyone is free to copy, modify, publish, use, compile, sell, or
  distribute this software, either in source code form or as a compiled
  binary, for any purpose, commercial or non-commercial, and by any
  means.

  In jurisdictions that recognize copyright laws, the author or authors
  of this software dedicate any and all copyright interest in the
  software to the public domain. We make this dedication for the benefit
  of the public at large and to the detriment of our heirs and
  successors. We intend this dedication to be an overt act of
  relinquishment in perpetuity of all present and future rights to this
  software under copyright law.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.

  For more information, please refer to <http://unlicense.org/>
*/
