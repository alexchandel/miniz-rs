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

#![feature(macro_rules, slicing_syntax, globs, unboxed_closures, if_let, asm, fn_traits)]
#![crate_type = "lib"]
#![allow(non_camel_case_types, non_snake_case)]

extern crate libc;
#[macro_use]
extern crate bitflags;

mod tdefl;
mod tinfl;

trait SizeOf {
    fn size_of(&self) -> usize;
}

impl <T> SizeOf for T {
    fn size_of(&self) -> usize {
        use std::mem::{size_of};
        return size_of::<T>();
    }
}

// ------------------- zlib-style API Definitions.

// For more compatibility with zlib, miniz.c uses unsigned long for some parameters/struct members. Beware: mz_ulong can be either 32 or 64-bits!
// type mz_ulong = libc::c_ulong;

const MZ_ADLER32_INIT: u32 = (1);

const MZ_CRC32_INIT: u32 = (0);

// Compression strategies.
enum CompressionStrategies { MZ_DEFAULT_STRATEGY = 0, MZ_FILTERED = 1, MZ_HUFFMAN_ONLY = 2, MZ_RLE = 3, MZ_FIXED = 4 }

// Method
const MZ_DEFLATED: isize = 8;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MINIZ_USE_UNALIGNED_LOADS_AND_STORES: usize = 1;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const MINIZ_USE_UNALIGNED_LOADS_AND_STORES: usize = 0;

// ------------------- End of Header: Implementation follows. (If you only want the header, define MINIZ_HEADER_FILE_ONLY.)

// ------------------- zlib-style API's

/// Adler-32 checksum algorithm
/// mz_adler32() returns the initial adler-32 value to use when called with ptr==NULL.
fn mz_adler32(adler: u32, buf: Option<&[u8]>) -> u32
{
  let mut buf: &[u8] = match buf { Some(x) => x, None => return MZ_ADLER32_INIT };
  let mut s1: u32 = (adler & 0xffff) as u32;
  let mut s2: u32 = (adler >> 16) as u32;
  let mut block_len: usize = buf.len() % 5552;
  while buf.len() > 0 {
    for i in buf[..block_len].iter() {
      s1 += *i as u32; s2 += s1;
    }
    s1 %= 65521u32; s2 %= 65521u32; buf = &buf[block_len..]; block_len = 5552usize;
  }
  return ((s2 << 16) + s1) as u32;
}

/// Karl Malbrain's compact CRC-32. See "A compact CCITT crc16 and crc32 C implementation that balances processor cache usage against speed": http://www.geocities.com/malbrain/
/// mz_crc32() returns the initial CRC-32 value to use when called with ptr==NULL.
fn mz_crc32(crc: u32, buf: Option<&[u8]>) -> u32
{
  let mut buf: &[u8] = match buf { Some(x) => x, None => return MZ_CRC32_INIT };
  let s_crc32: [u32; 16] = [ 0, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4, 0x4db26158, 0x5005713c,
    0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c, 0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c ];
  let mut crcu32: u32 = crc as u32;
  crcu32 = !crcu32;
  for b in buf.iter() {
    crcu32 = (crcu32 >> 4) ^ s_crc32[((crcu32 as u8 & 0xF) ^ (*b & 0xF)) as usize];
    crcu32 = (crcu32 >> 4) ^ s_crc32[((crcu32 as u8 & 0xF) ^ (*b >> 4)) as usize];
  }
  return !crcu32;
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
