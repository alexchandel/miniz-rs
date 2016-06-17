use libc::{c_int, c_uint};
use std::slice::from_raw_parts;
use {mz_adler32, SizeOf};

use tinfl::tinfl_status::*;

// ------------------- Low-level Decompression API Definitions

/// Decompression flags used by tinfl_decompress().
/// TINFL_FLAG_PARSE_ZLIB_HEADER: If set, the input has a valid zlib header and ends with an adler32 checksum (it's a valid zlib stream). Otherwise, the input is a raw deflate stream.
/// TINFL_FLAG_HAS_MORE_INPUT: If set, there are more input bytes available beyond the end of the supplied input buffer. If clear, the input buffer contains all remaining input.
/// TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF: If set, the output buffer is large enough to hold the entire decompressed stream. If clear, the output buffer is at least the size of the dictionary (typically 32KB).
/// TINFL_FLAG_COMPUTE_ADLER32: Force adler-32 checksum computation of the decompressed bytes.
bitflags! {
  pub flags DecompressionFlags: u32 {
    const TINFL_FLAG_PARSE_ZLIB_HEADER = 1,
    const TINFL_FLAG_HAS_MORE_INPUT = 2,
    const TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF = 4,
    const TINFL_FLAG_COMPUTE_ADLER32 = 8
  }
}

const TINFL_DECOMPRESS_MEM_TO_MEM_FAILED: usize = !0 as usize;

pub type tinfl_put_buf_func_ptr<'a> = &'a mut FnMut<(*const u8, usize), Output = bool>;

/// Max size of LZ dictionary.
const TINFL_LZ_DICT_SIZE: usize = 32768;

/// Return status.
#[repr(i8)]
#[derive(PartialEq, Clone, Copy)]
enum tinfl_status
{
  TINFL_STATUS_BAD_PARAM = -3,
  TINFL_STATUS_ADLER32_MISMATCH = -2,
  TINFL_STATUS_FAILED = -1,
  TINFL_STATUS_DONE = 0,
  TINFL_STATUS_NEEDS_MORE_INPUT = 1,
  TINFL_STATUS_HAS_MORE_OUTPUT = 2
}

fn tinfl_get_adler32(r: &tinfl_decompressor) -> u32 {
  r.m_check_adler32
}

// Internal/private bits follow.
const TINFL_MAX_HUFF_TABLES: usize = 3;
const TINFL_MAX_HUFF_SYMBOLS_0: usize = 288;
const TINFL_MAX_HUFF_SYMBOLS_1: usize = 32;
const TINFL_MAX_HUFF_SYMBOLS_2: usize = 19;
const TINFL_FAST_LOOKUP_BITS: usize = 10;
const TINFL_FAST_LOOKUP_SIZE: usize = 1 << TINFL_FAST_LOOKUP_BITS;

struct tinfl_huff_table
{
  m_code_size: [u8; TINFL_MAX_HUFF_SYMBOLS_0],
  m_look_up: [i16; TINFL_FAST_LOOKUP_SIZE],
  m_tree: [i16; TINFL_MAX_HUFF_SYMBOLS_0 * 2]
}

impl tinfl_huff_table {
  fn new() -> tinfl_huff_table {
    tinfl_huff_table {
      m_code_size: [0u8; TINFL_MAX_HUFF_SYMBOLS_0],
      m_look_up: [0i16; TINFL_FAST_LOOKUP_SIZE],
      m_tree: [0i16; TINFL_MAX_HUFF_SYMBOLS_0 * 2]
    }
  }
}

#[cfg(target_word_size = "64")]
type tinfl_bit_buf_t = u64;
#[cfg(target_word_size = "64")]
const TINFL_BITBUF_SIZE: usize = (64);
#[cfg(not(target_word_size = "64"))]
type tinfl_bit_buf_t = u32;
#[cfg(not(target_word_size = "64"))]
const TINFL_BITBUF_SIZE: usize = (32);

struct tinfl_decompressor
{
  m_state: u32, m_num_bits: usize, m_zhdr0: u32, m_zhdr1: u32, m_z_adler32: u32, m_final: u32, m_type: usize,
  m_check_adler32: u32, m_dist: usize, m_counter: usize, m_num_extra: usize, m_table_sizes: [usize; TINFL_MAX_HUFF_TABLES],
  m_bit_buf: tinfl_bit_buf_t,
  m_dist_from_out_buf_start: usize,
  m_tables: [tinfl_huff_table; TINFL_MAX_HUFF_TABLES],
  m_raw_header: [u8; 4], m_len_codes: [u8; TINFL_MAX_HUFF_SYMBOLS_0 + TINFL_MAX_HUFF_SYMBOLS_1 + 137]
}

impl tinfl_decompressor {
  /// Initializes the decompressor to its initial state.
  fn new() -> tinfl_decompressor {
    tinfl_decompressor {
      m_state: 0u32, m_num_bits: 0usize, m_zhdr0: 0u32, m_zhdr1: 0u32, m_z_adler32: 0u32, m_final: 0u32, m_type: 0usize,
      m_check_adler32: 0u32, m_dist: 0usize, m_counter: 0usize, m_num_extra: 0usize, m_table_sizes: [0usize; TINFL_MAX_HUFF_TABLES],
      m_bit_buf: 0,
      m_dist_from_out_buf_start: 0usize,
      //m_tables should be  TINFL_MAX_HUFF_TABLES long, but huff_table can't be copied
      m_tables: [tinfl_huff_table::new(), tinfl_huff_table::new(), tinfl_huff_table::new()],
      m_raw_header: [0u8; 4], m_len_codes: [0u8; TINFL_MAX_HUFF_SYMBOLS_0 + TINFL_MAX_HUFF_SYMBOLS_1 + 137]
    }
  }
}

// ------------------- Low-level Decompression (completely independent from all compression API's)

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
macro_rules! MZ_READ_LE16( ($p:expr) => (*(($p) as *const u16)); );
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little"))]
macro_rules! MZ_READ_LE32( ($p:expr) => (*(($p) as *const u32)); );

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little")))]
macro_rules! MZ_READ_LE16( ($p:expr) => ((u32)((($p) as *const u8)[0]) | ((u32)((($p) as *const u8)[1]) << 8u)) );
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_endian = "little")))]
macro_rules! MZ_READ_LE32( ($p:expr) => ((u32)((($p) as *const u8)[0]) | ((u32)((($p) as *const u8)[1]) << 8u) | ((u32)((($p) as *const u8)[2]) << 16u) | ((u32)((($p) as *const u8)[3]) << 24u)) );

// #define TINFL_CR_BEGIN switch(r->m_state) { case 0:
// #define TINFL_CR_FINISH }

/// Main low-level decompressor coroutine function. This is the only function actually needed for decompression. All the other functions are just high-level helpers for improved usability.
/// This is a universal API, i.e. it can be used as a building block to build any desired higher level decompression API. In the limit case, it can be called once per every byte input or output.
unsafe fn tinfl_decompress(r: &mut tinfl_decompressor, pIn_buf_next: *const u8, pIn_buf_size: &mut usize, pOut_buf_start: *mut u8, pOut_buf_next: *mut u8, pOut_buf_size: &mut usize, decomp_flags: DecompressionFlags) -> tinfl_status
{
  let s_length_base: [usize; 31] = [ 3,4,5,6,7,8,9,10,11,13, 15,17,19,23,27,31,35,43,51,59, 67,83,99,115,131,163,195,227,258,0,0 ];
  let s_length_extra: [isize; 31]= [ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0 ];
  let s_dist_base: [isize; 32] = [ 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193, 257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0];
  let s_dist_extra: [isize; 32] = [ 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13, 0xFFFF, 0xFFFF ]; // WARNING miniz.c HAD WRONG INITIALIZER
  let s_length_dezigzag: [usize; 19] = [ 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 ];
  let s_min_table_sizes: [usize; 3] = [ 257, 1, 4 ];

  let mut status: tinfl_status = TINFL_STATUS_FAILED; let num_bits: usize; let dist: usize; let counter: usize; let num_extra: usize; let bit_buf: tinfl_bit_buf_t;
  let pIn_buf_cur: *const u8 = pIn_buf_next; let pIn_buf_end: *const u8 = pIn_buf_next.offset(*pIn_buf_size as isize);
  let pOut_buf_cur: *mut u8 = pOut_buf_next; let pOut_buf_end:  *mut u8 = pOut_buf_next.offset(*pOut_buf_size as isize);
  let out_buf_size_mask: usize = if decomp_flags.contains(TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF) {!0 as usize} else {((pOut_buf_next as usize - pOut_buf_start as usize) + *pOut_buf_size) - 1};
  let dist_from_out_buf_start: usize;

  macro_rules! TINFL_CR_BEGIN(
    ($i: expr) => ({
      if r.m_state == $i {
        asm!(concat!("jmp ", " case_", stringify!($i)) ::::"volatile");
      }
    });
  );

  macro_rules! TINFL_CR_RETURN(
    ($state_index: expr, $result: expr) => ({
        status = $result;
        r.m_state = $state_index;
        asm!("jmp common_exit" :::: "volatile");
        asm!(concat!("case_", stringify!($state_index)) :::: "volatile");
      };
    );
  );
  macro_rules! TINFL_CR_RETURN_FOREVER( ($state_index:expr, $result:expr) => ( { loop { TINFL_CR_RETURN!($state_index, $result); } }; ); );

  // TODO: If the caller has indicated that there's no more input, and we attempt to read beyond the input buf, then something is wrong with the input because the inflator never
  // reads ahead more than it needs to. Currently TINFL_GET_BYTE() pads the end of the stream with 0's in this scenario.
  macro_rules! TINFL_GET_BYTE( ($state_index:expr, $c:expr) => (loop {
    if (pIn_buf_cur >= pIn_buf_end) {
      loop {
        if decomp_flags.contains(TINFL_FLAG_HAS_MORE_INPUT) {
          TINFL_CR_RETURN!($state_index, TINFL_STATUS_NEEDS_MORE_INPUT);
          if (pIn_buf_cur < pIn_buf_end) {
            $c = *pIn_buf_cur;
            pIn_buf_cur = pIn_buf_cur.offset(1);
            break;
          }
        } else {
          $c = 0;
          break;
        }
      }
    } else {$c = *pIn_buf_cur; pIn_buf_cur = pIn_buf_cur.offset(1);} break; }; );
  );

  macro_rules! TINFL_NEED_BITS( ($state_index:expr, $n:expr) => (loop { let c: usize; TINFL_GET_BYTE!($state_index, c); bit_buf |= ((c as tinfl_bit_buf_t) << num_bits); num_bits += 8; if !(num_bits < ($n)) {break;} }); );
  macro_rules! TINFL_SKIP_BITS( ($state_index:expr, $n:expr) => ({ if (num_bits < ($n)) { TINFL_NEED_BITS!($state_index, $n); } bit_buf >>= ($n); num_bits -= ($n); }); );
  macro_rules! TINFL_GET_BITS( ($state_index:expr, $b:expr, $n:expr) => (
    { if num_bits < ($n) { TINFL_NEED_BITS!($state_index, $n); } $b = bit_buf & ((1 << ($n)) - 1); bit_buf >>= ($n); num_bits -= ($n); });
  );

  let mut huff_decode_temp: c_int;
  let mut huff_decode_code_len: usize;
  let mut huff_decode_c: usize;

  // TINFL_HUFF_BITBUF_FILL() is only used rarely, when the number of bytes remaining in the input buffer falls below 2.
  // It reads just enough bytes from the input stream that are needed to decode the next Huffman code (and absolutely no more). It works by trying to fully decode a
  // Huffman code by using whatever bits are currently present in the bit buffer. If this fails, it reads another byte, and tries again until it succeeds or until the
  // bit buffer contains >=15 bits (deflate's max. Huffman code size).
  macro_rules! TINFL_HUFF_BITBUF_FILL( (
    $state_index:expr, $pHuff:expr) => (
    loop {
      huff_decode_temp = ($pHuff).m_look_up[bit_buf as usize & (TINFL_FAST_LOOKUP_SIZE - 1)];
      if (huff_decode_temp >= 0) {
        huff_decode_code_len = huff_decode_temp >> 9;
        if (huff_decode_code_len > 0) && (num_bits >= huff_decode_code_len) {
          break;
        }
      } else if (num_bits > TINFL_FAST_LOOKUP_BITS) {
         huff_decode_code_len = TINFL_FAST_LOOKUP_BITS;
         loop {
            huff_decode_temp = ($pHuff).m_tree[!huff_decode_temp + ((bit_buf >> {let t = huff_decode_code_len; huff_decode_code_len+=1; t}) & 1)];
            if !((huff_decode_temp < 0) && (num_bits >= (huff_decode_code_len + 1))) {break;}
         };
         if huff_decode_temp >= 0 {break;}
      } TINFL_GET_BYTE!($state_index, huff_decode_c); bit_buf |= ((huff_decode_c as tinfl_bit_buf_t) << num_bits); num_bits += 8;
      if !(num_bits < 15) {break;};
    });
  );

  // TINFL_HUFF_DECODE() decodes the next Huffman coded symbol. It's more complex than you would initially expect because the zlib API expects the decompressor to never read
  // beyond the final byte of the deflate stream. (In other words, when this macro wants to read another byte from the input, it REALLY needs another byte in order to fully
  // decode the next Huffman code.) Handling this properly is particularly important on raw deflate (non-zlib) streams, which aren't followed by a byte aligned adler-32.
  // The slow path is only executed at the very end of the input buffer.
  macro_rules! TINFL_HUFF_DECODE( ($state_index:expr, $sym:expr, $pHuff:expr) => (loop {
    // let mut huff_decode_temp: c_int; let mut huff_decode_code_len: usize; let mut huff_decode_c: usize;
    if (num_bits < 15) {
      if ((pIn_buf_end as usize - pIn_buf_cur as usize) < 2) {
         TINFL_HUFF_BITBUF_FILL!($state_index, $pHuff);
      } else {
         bit_buf |= ((*pIn_buf_cur as tinfl_bit_buf_t) << num_bits) | (( *pIn_buf_cur.offset(1) as tinfl_bit_buf_t) << (num_bits + 8));
         pIn_buf_cur = pIn_buf_cur.offset(2);
         num_bits += 16;
      }
    }
    huff_decode_temp = ($pHuff).m_look_up[bit_buf as usize & (TINFL_FAST_LOOKUP_SIZE - 1)];
    if huff_decode_temp >= 0 {
      huff_decode_code_len = huff_decode_temp >> 9; huff_decode_temp &= 511;
    } else {
      huff_decode_code_len = TINFL_FAST_LOOKUP_BITS;
      loop {
        huff_decode_temp = ($pHuff).m_tree[!huff_decode_temp + ((bit_buf >> {let t = huff_decode_code_len; huff_decode_code_len+=1; t}) & 1)];
        if !(huff_decode_temp < 0) {break;}
      }
    } $sym = huff_decode_temp; bit_buf >>= huff_decode_code_len; num_bits -= huff_decode_code_len; break;}; );
  );

  // Ensure the output buffer's size is a power of 2, unless the output buffer is large enough to hold the entire output file (in which case it doesn't matter).
  if ((out_buf_size_mask + 1) & out_buf_size_mask != 0) || (pOut_buf_next < pOut_buf_start) { *pIn_buf_size = 0; *pOut_buf_size = 0; return TINFL_STATUS_BAD_PARAM; }

  num_bits = r.m_num_bits; bit_buf = r.m_bit_buf; dist = r.m_dist; counter = r.m_counter; num_extra = r.m_num_extra; dist_from_out_buf_start = r.m_dist_from_out_buf_start;
  // TINFL_CR_BEGIN
  {
    /*
    TINFL_CR_BEGIN!(0);
    TINFL_CR_BEGIN!(1);
    TINFL_CR_BEGIN!(2);
    TINFL_CR_BEGIN!(3);
    TINFL_CR_BEGIN!(4);
    TINFL_CR_BEGIN!(5);
    TINFL_CR_BEGIN!(6);
    TINFL_CR_BEGIN!(7);
    TINFL_CR_BEGIN!(8);
    TINFL_CR_BEGIN!(9);
    TINFL_CR_BEGIN!(10);
    TINFL_CR_BEGIN!(11);
    TINFL_CR_BEGIN!(12);
    TINFL_CR_BEGIN!(13);
    TINFL_CR_BEGIN!(14);
    TINFL_CR_BEGIN!(15);
    TINFL_CR_BEGIN!(16);
    TINFL_CR_BEGIN!(17);
    TINFL_CR_BEGIN!(18);
    TINFL_CR_BEGIN!(19);
    TINFL_CR_BEGIN!(20);
    TINFL_CR_BEGIN!(21);
    TINFL_CR_BEGIN!(22);
    TINFL_CR_BEGIN!(23);
    TINFL_CR_BEGIN!(24);
    TINFL_CR_BEGIN!(25);
    TINFL_CR_BEGIN!(26);
    TINFL_CR_BEGIN!(27);
    TINFL_CR_BEGIN!(28);
    TINFL_CR_BEGIN!(29);
    TINFL_CR_BEGIN!(30);
    TINFL_CR_BEGIN!(31);
    TINFL_CR_BEGIN!(32);
    TINFL_CR_BEGIN!(33);
    TINFL_CR_BEGIN!(34);
    TINFL_CR_BEGIN!(35);
    TINFL_CR_BEGIN!(36);
    TINFL_CR_BEGIN!(37);
    TINFL_CR_BEGIN!(38);
    TINFL_CR_BEGIN!(39);
    TINFL_CR_BEGIN!(40);
    TINFL_CR_BEGIN!(41);
    TINFL_CR_BEGIN!(42);
    TINFL_CR_BEGIN!(43);
    TINFL_CR_BEGIN!(44);
    TINFL_CR_BEGIN!(45);
    TINFL_CR_BEGIN!(46);
    TINFL_CR_BEGIN!(47);
    TINFL_CR_BEGIN!(48);
    TINFL_CR_BEGIN!(49);
    TINFL_CR_BEGIN!(50);
    TINFL_CR_BEGIN!(51);
    TINFL_CR_BEGIN!(52);
    TINFL_CR_BEGIN!(53);
    TINFL_CR_BEGIN!(54);
    TINFL_CR_BEGIN!(55);
    TINFL_CR_BEGIN!(56);
    TINFL_CR_BEGIN!(57);
    TINFL_CR_BEGIN!(58);
    TINFL_CR_BEGIN!(59);
    TINFL_CR_BEGIN!(60);
    TINFL_CR_BEGIN!(61);
    TINFL_CR_BEGIN!(62);
    TINFL_CR_BEGIN!(63);
    if r.m_state >= 64 {panic!();}

    asm!("case_0:"::::"volatile");
    bit_buf = 0;
    num_bits = 0;
    dist = 0;
    counter = 0;
    num_extra = 0;
    r.m_zhdr0 = 0;
    r.m_zhdr1 = 0;
    r.m_z_adler32 = 1;
    r.m_check_adler32 = 1;
    if decomp_flags.contains(TINFL_FLAG_PARSE_ZLIB_HEADER)
    {
      TINFL_GET_BYTE!(1, r.m_zhdr0); TINFL_GET_BYTE!(2, r.m_zhdr1);
      counter = (((r.m_zhdr0 * 256 + r.m_zhdr1) % 31 != 0) || (r.m_zhdr1 & 32 != 0) || ((r.m_zhdr0 & 15) != 8)) as usize;
      if !decomp_flags.contains(TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF) {
        counter |= (((1u << (8u + (r.m_zhdr0 >> 4) as usize)) > 32768u) || ((out_buf_size_mask + 1) < (1u << (8u + (r.m_zhdr0 >> 4) as usize)))) as usize
      };
      if counter > 0 { TINFL_CR_RETURN_FOREVER!(36, TINFL_STATUS_FAILED); }
    }
    loop
    {
      TINFL_GET_BITS!(3, r.m_final, 3); r.m_type = r.m_final >> 1;
      if (r.m_type == 0)
      {
        TINFL_SKIP_BITS!(5, num_bits & 7);
        counter = 0;
        while counter < 4 { if num_bits > 0 {TINFL_GET_BITS!(6, r.m_raw_header[counter], 8);} else {TINFL_GET_BYTE!(7, r.m_raw_header[counter]);}; counter+=1}

        counter = r.m_raw_header[0] as usize | (r.m_raw_header[1] as usize << 8);
        if (counter != (0xFFFF ^ (r.m_raw_header[2] | (r.m_raw_header[3] << 8))) as usize) { TINFL_CR_RETURN_FOREVER!(39, TINFL_STATUS_FAILED); }
        while (counter > 0) && (num_bits > 0)
        {
          TINFL_GET_BITS!(51, dist, 8);
          while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN!(52, TINFL_STATUS_HAS_MORE_OUTPUT); }
          *pOut_buf_cur = dist as u8; pOut_buf_cur = pOut_buf_cur.offset(1);
          counter-=1;
        }
        while counter>0
        {
          let n: usize;
          while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN!(9, TINFL_STATUS_HAS_MORE_OUTPUT); }
          while (pIn_buf_cur >= pIn_buf_end)
          {
            if decomp_flags.contains(TINFL_FLAG_HAS_MORE_INPUT)
            {
              TINFL_CR_RETURN!(38, TINFL_STATUS_NEEDS_MORE_INPUT);
            }
            else
            {
              TINFL_CR_RETURN_FOREVER!(40, TINFL_STATUS_FAILED);
            }
          }
          n = min(min(pOut_buf_end as usize - pOut_buf_cur as usize, pIn_buf_end as usize - pIn_buf_cur as usize), counter);
          copy_memory(pOut_buf_cur, pIn_buf_cur, n);
          pIn_buf_cur = pIn_buf_cur.offset(n as isize);
          pOut_buf_cur = pOut_buf_cur.offset(n as isize);
          counter -= n;
        }
      }
      else if (r.m_type == 3)
      {
        TINFL_CR_RETURN_FOREVER!(10, TINFL_STATUS_FAILED);
      }
      else
      {
        if (r.m_type == 1)
        {
          let mut p: *mut u8 = r.m_tables[0].m_code_size.as_mut_ptr(); let i: usize;
          r.m_table_sizes[0] = 288; r.m_table_sizes[1] = 32; for i in r.m_tables[1].m_code_size[mut 0..32].iter_mut() {*i = 5;}
          i = 0;
          while i <= 143 {*p = 8; p = p.offset(1); i+=1;}
          while i <= 255 {*p = 9; p = p.offset(1); i+=1;}
          while i <= 279 {*p = 7; p = p.offset(1); i+=1;}
          while i <= 287 {*p = 8; p = p.offset(1); i+=1;}
        }
        else
        {
          counter = 0;
          while counter < 3 { TINFL_GET_BITS!(11, r.m_table_sizes[counter], [5,5,4][counter]); r.m_table_sizes[counter] += s_min_table_sizes[counter]; counter+=1; }
          for i in r.m_tables[2].m_code_size.iter_mut() {*i = 0;};
          counter = 0; while counter < r.m_table_sizes[2] { let s: usize; TINFL_GET_BITS!(14, s, 3); r.m_tables[2].m_code_size[s_length_dezigzag[counter]] = s as u8; counter+=1; }
          r.m_table_sizes[2] = 19;
        }
        while r.m_type as isize >= 0
        {
          let i: usize; let j: usize;
          let next_code = [0u, ..17]; let total_syms = [0u, ..16];
          let pTable: &mut tinfl_huff_table = &r.m_tables[r.m_type]; for i in pTable.m_look_up.iter_mut() {*i=0;}; for i in pTable.m_tree.iter_mut() {*i=0;};
          i = 0;
          while i < r.m_table_sizes[r.m_type] {total_syms[pTable.m_code_size[i] as usize] += 1; i += 1;}
          let used_syms: usize = 0; let total: usize = 0; next_code[0] = 0; next_code[1] = 0;
          i = 1;
          while i <= 15 {
            used_syms += total_syms[i];
            total = ((total + total_syms[i]) << 1);
            next_code[i + 1] = total;
            i += 1;
          }
          if ((65536 != total) && (used_syms > 1))
          {
            TINFL_CR_RETURN_FOREVER!(35, TINFL_STATUS_FAILED);
          }
          let tree_next: isize = -1; let tree_cur: isize; let sym_index: usize = 0;
          while sym_index < r.m_table_sizes[r.m_type]
          {
            let rev_code: usize = 0;
            let l: usize;
            let cur_code: usize;
            let code_size: usize = pTable.m_code_size[sym_index] as usize; if code_size==0 {continue;}
            next_code[code_size]+=1;
            cur_code = next_code[code_size];
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
              rev_code >>= 1;
              tree_cur -= rev_code & 1;
              if (!pTable.m_tree[-tree_cur - 1]) { pTable.m_tree[(-tree_cur - 1) as usize] = tree_next as i16; tree_cur = tree_next; tree_next -= 2; } else {tree_cur = pTable.m_tree[-tree_cur - 1];}
              j -= 1;
            }
            tree_cur -= ((rev_code >>= 1) & 1); pTable.m_tree[(-tree_cur - 1) as usize] = sym_index as i16;
            sym_index += 1;
          }
          if (r.m_type == 2)
          {
            counter = 0;
            while counter < (r.m_table_sizes[0] + r.m_table_sizes[1])
            {
              let s: c_uint;
              TINFL_HUFF_DECODE!(16, dist, &r.m_tables[2]); if (dist < 16) { r.m_len_codes[counter] = dist as u8; counter+=1; continue; }
              if (dist == 16) && (counter == 0)
              {
                TINFL_CR_RETURN_FOREVER!(17, TINFL_STATUS_FAILED);
              }
              num_extra = [2,3,7][dist - 16]; TINFL_GET_BITS!(18, s, num_extra); s += [3,3,13][dist - 16];
              for i in r.m_len_codes[mut counter..counter+s].iter_mut() {*i = if dist == 16 {r.m_len_codes[counter - 1]} else {0};}
              // set_memory(r.m_len_codes + counter, if dist == 16 {r.m_len_codes[counter - 1]} else {0}, s);
              counter += s;
            }
            if ((r.m_table_sizes[0] + r.m_table_sizes[1]) != counter)
            {
              TINFL_CR_RETURN_FOREVER!(21, TINFL_STATUS_FAILED);
            }
            copy_memory(r.m_tables[0].m_code_size.as_mut_ptr(), r.m_len_codes.as_ptr(), r.m_table_sizes[0]);
            copy_memory(r.m_tables[1].m_code_size.as_mut_ptr(), r.m_len_codes.as_ptr().offset(r.m_table_sizes[0]), r.m_table_sizes[1]);
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
              TINFL_HUFF_DECODE!(23, counter, &r.m_tables[0]);
              if (counter >= 256){
                break;
              }
              while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN!(24, TINFL_STATUS_HAS_MORE_OUTPUT); }
              *pOut_buf_cur = counter as u8;
              pOut_buf_cur += 1;
            }
            else
            {
              let sym2: usize; let code_len: usize;

              match TINFL_BITBUF_SIZE {
                32 => {if (num_bits < 15) { bit_buf |= ((MZ_READ_LE16!(pIn_buf_cur) as tinfl_bit_buf_t) << num_bits); pIn_buf_cur = pIn_buf_cur.offset(2); num_bits += 16; }},
                64 => {if (num_bits < 30) { bit_buf |= ((MZ_READ_LE32!(pIn_buf_cur) as tinfl_bit_buf_t) << num_bits); pIn_buf_cur = pIn_buf_cur.offset(4); num_bits += 32; }},
                _ => panic!(), // 32 is probably OK for 16-bit too
              }

              sym2 = r.m_tables[0].m_look_up[bit_buf as usize & (TINFL_FAST_LOOKUP_SIZE - 1)];
              if sym2 >= 0 {
                code_len = sym2 >> 9;
              } else {
                code_len = TINFL_FAST_LOOKUP_BITS; loop { sym2 = r.m_tables[0].m_tree[!sym2 + ((bit_buf >> code_len) & 1)]; code_len += 1; if !(sym2 < 0) {break;} };
              }
              counter = sym2; bit_buf >>= code_len; num_bits -= code_len;
              if counter & 256 > 0{
                break;
              }

              match TINFL_BITBUF_SIZE {
                32 => {},
                64 => {if (num_bits < 15) { bit_buf |= ((MZ_READ_LE16!(pIn_buf_cur) as tinfl_bit_buf_t) << num_bits); pIn_buf_cur = pIn_buf_cur.offset(2); num_bits += 16; }},
                _ => panic!(), // 32 is probably OK for 16-bit too
              }

              sym2 = r.m_tables[0].m_look_up[bit_buf as usize & (TINFL_FAST_LOOKUP_SIZE - 1)];
              if sym2 >= 0 {
                code_len = sym2 >> 9;
              } else {
                code_len = TINFL_FAST_LOOKUP_BITS; loop { sym2 = r.m_tables[0].m_tree[!sym2 + ((bit_buf >> code_len) & 1)]; code_len += 1; if !(sym2 < 0) {break;} };
              }
              bit_buf >>= code_len; num_bits -= code_len;

              *pOut_buf_cur = counter as u8;
              if (sym2 & 256)
              {
                pOut_buf_cur = pOut_buf_cur.offset(1);
                counter = sym2;
                break;
              }
              *pOut_buf_cur.offset(1) = sym2 as u8;
              pOut_buf_cur = pOut_buf_cur.offset(2);
            }
          }
          if ((counter &= 511) == 256) {break;}

          num_extra = s_length_extra[counter - 257]; counter = s_length_base[counter - 257];
          if num_extra { let extra_bits: c_uint; TINFL_GET_BITS!(25, extra_bits, num_extra); counter += extra_bits; }

          TINFL_HUFF_DECODE!(26, dist, &r.m_tables[1]);
          num_extra = s_dist_extra[dist]; dist = s_dist_base[dist];
          if num_extra { let extra_bits: c_uint; TINFL_GET_BITS!(27, extra_bits, num_extra); dist += extra_bits; }

          dist_from_out_buf_start = pOut_buf_cur - pOut_buf_start;
          if ((dist > dist_from_out_buf_start) && (decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF))
          {
            TINFL_CR_RETURN_FOREVER!(37, TINFL_STATUS_FAILED);
          }

          pSrc = pOut_buf_start + ((dist_from_out_buf_start - dist) & out_buf_size_mask);

          if (max(pOut_buf_cur as usize, pSrc as usize) + counter) > pOut_buf_end as usize
          {
            while {let _temp = counter; counter-=1; _temp}
            {
              while (pOut_buf_cur >= pOut_buf_end) { TINFL_CR_RETURN!(53, TINFL_STATUS_HAS_MORE_OUTPUT); }
              *pOut_buf_cur = pOut_buf_start[(dist_from_out_buf_start - dist) & out_buf_size_mask];
              dist_from_out_buf_start += 1;
              pOut_buf_cur += 1;
            }
            continue;
          }
          else if ((counter >= 9) && (counter <= dist))
          {
            match MINIZ_USE_UNALIGNED_LOADS_AND_STORES {
              1 => {
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
              },
              0 => {}
            }
          }
          loop
          {
            *pOut_buf_cur.offset(0) = *pSrc.offset(0);
            *pOut_buf_cur.offset(1) = *pSrc.offset(1);
            *pOut_buf_cur.offset(2) = *pSrc.offset(2);
            pOut_buf_cur = pOut_buf_cur.offset(3); pSrc = pSrc.offset();
            if !((counter -= 3) as isize > 2) {break;}
          }
          if (counter as isize > 0)
          {
            pOut_buf_cur[0] = pSrc[0];
            if (counter as isize > 1){
              pOut_buf_cur[1] = pSrc[1];
            }
            pOut_buf_cur = pOut_buf_cur.offset(counter);
          }
        }
      }
      if (r.m_final & 1) > 0 {break;}
    }
    if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER)
    {
      TINFL_SKIP_BITS!(32, num_bits & 7);
      counter = 0;
      while counter < 4 {
        let s: c_uint;
        if num_bits > 0 {TINFL_GET_BITS!(41, s, 8);} else {TINFL_GET_BYTE!(42, s);};
        r.m_z_adler32 = (r.m_z_adler32 << 8) | s;
        counter+=1;
      }
    }
    TINFL_CR_RETURN_FOREVER!(34, TINFL_STATUS_DONE); // */
  // TINFL_CR_FINISH
  }

  asm!("common_exit:" :::: "volatile");
  {
    r.m_num_bits = num_bits; r.m_bit_buf = bit_buf; r.m_dist = dist; r.m_counter = counter; r.m_num_extra = num_extra; r.m_dist_from_out_buf_start = dist_from_out_buf_start;
    *pIn_buf_size = pIn_buf_cur as usize - pIn_buf_next as usize;
    *pOut_buf_size = pOut_buf_cur as usize - pOut_buf_next as usize;
    if decomp_flags.contains(TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_COMPUTE_ADLER32) && (status as i8 >= 0)
    {
      r.m_check_adler32 = mz_adler32(r.m_check_adler32, 
        Some(from_raw_parts(pOut_buf_next as *const u8, *pOut_buf_size))
      );

      if (status == TINFL_STATUS_DONE) && decomp_flags.contains(TINFL_FLAG_PARSE_ZLIB_HEADER) && (r.m_check_adler32 != r.m_z_adler32) {status = TINFL_STATUS_ADLER32_MISMATCH;};
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
  let mut decomp = tinfl_decompressor::new();
  // Create output buffer.
  // WARNING: we no longer pass a NULL pointer to tinfl_decompress
  // on the first pass (as miniz.c did). TODO: ensure this decompresses correctly.
  let mut out_buf = vec![0u8; 128];
  // Track position in input buffer
  let mut src_buf_ofs: usize = 0;
  // Track position in output buffer
  let mut out_buf_ofs: usize = 0;
  loop {
    let mut src_buf_size: usize = src_buf.len() - src_buf_ofs;
    let mut out_buf_size: usize = out_buf.len() - out_buf_ofs;
    let status: tinfl_status = unsafe {tinfl_decompress(
      &mut decomp,
      src_buf[src_buf_ofs..].as_ptr(),
      &mut src_buf_size,
      out_buf[..].as_mut_ptr(),
      out_buf[out_buf_ofs..].as_mut_ptr(),
      &mut out_buf_size,
      (flags & !TINFL_FLAG_HAS_MORE_INPUT) | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF
    )};

    if (status as i8) < 0 || (status == TINFL_STATUS_NEEDS_MORE_INPUT) {return None;}
    // now, status is either TINFL_STATUS_HAS_MORE_OUTPUT or TINFL_STATUS_DONE

    // Increase buffer offsets to reflect newly copied data.
    // tinfl_decompress writes this back into src_buf_size and out_buf_size
    src_buf_ofs += src_buf_size;
    out_buf_ofs += out_buf_size;

    // If all data is copied, end.
    if status == TINFL_STATUS_DONE {break;}
    // Otherwise, double the output buffer capacity & length.
    {let _borrow_temp = out_buf.len(); out_buf.extend(::std::iter::repeat(0u8).take(_borrow_temp))};
  }
  // Set length of output buffer to number of bytes copied, instead of capacity.
  unsafe {out_buf.set_len(out_buf_ofs)};
  Some(out_buf)
}

// tinfl_decompress_mem_to_mem() decompresses a block in memory to another block in memory.
// Returns TINFL_DECOMPRESS_MEM_TO_MEM_FAILED on failure, or the number of bytes written on success.
pub fn tinfl_decompress_mem_to_mem(out_buf: &mut[u8], src_buf: &[u8], flags: DecompressionFlags) -> usize
{
  let mut decomp = tinfl_decompressor::new();
  let mut src_buf_len: usize = src_buf.len();
  let mut out_buf_len: usize = out_buf.len();
  let status: tinfl_status = unsafe{tinfl_decompress(
    &mut decomp,
    src_buf.as_ptr(),
    &mut src_buf_len,
    out_buf.as_mut_ptr(),
    out_buf.as_mut_ptr(),
    &mut out_buf_len,
    (flags & !TINFL_FLAG_HAS_MORE_INPUT) | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF
  )};
  if status != TINFL_STATUS_DONE {TINFL_DECOMPRESS_MEM_TO_MEM_FAILED} else {out_buf_len}
}

// tinfl_decompress_mem_to_callback() decompresses a block in memory to an internal 32KB buffer, and a user provided callback function will be called to flush the buffer.
// Returns 1 on success or 0 on failure.
pub fn tinfl_decompress_mem_to_callback(in_buf: &[u8], put_buf_func: tinfl_put_buf_func_ptr, flags: DecompressionFlags) -> (bool, usize)
{
  let mut decomp = tinfl_decompressor::new();
  let mut dict: Vec<u8> = vec![0u8; TINFL_LZ_DICT_SIZE];
  let mut in_buf_ofs: usize = 0;
  let mut dict_ofs: usize = 0;
  let mut result: bool = false;
  loop {
    let mut in_buf_size: usize = in_buf.len() - in_buf_ofs;
    let mut dst_buf_size: usize = TINFL_LZ_DICT_SIZE - dict_ofs;
    let status: tinfl_status = unsafe{tinfl_decompress(
      &mut decomp,
      in_buf[in_buf_ofs..].as_ptr(),
      &mut in_buf_size,
      dict[..].as_mut_ptr(),
      dict[dict_ofs..].as_mut_ptr(),
      &mut dst_buf_size,
      (flags & !(TINFL_FLAG_HAS_MORE_INPUT | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF))
    )};

    // Increase input buffer to reflect data copied out.
    in_buf_ofs += in_buf_size;

    if (dst_buf_size > 0) && ! put_buf_func.call_mut((dict[dict_ofs..dict_ofs+dst_buf_size].as_ptr(),dst_buf_size)) {break;}
    if status != TINFL_STATUS_HAS_MORE_OUTPUT {
      result = status == TINFL_STATUS_DONE;
      break;
    }
    dict_ofs = (dict_ofs + dst_buf_size) & (TINFL_LZ_DICT_SIZE - 1);
  }
  (result, in_buf_ofs)
}
