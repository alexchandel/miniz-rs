miniz-rs
========

A translation of the miniz.c single source Deflate/Inflate compression library to Rust.

Presently, Rust's libflate relies on miniz.c.
Even other compression libraries like [flate2-rs](https://github.com/alexcrichton/flate2-rs) rely on a statically linked miniz.
Having a pure-Rust compression implementation would simplify the implementation of such libraries.

## Progress
This does not compile yet.

Translation is still a WIP! The simplest translations such as function signatures, type declarations, and increment/decrement replacements have been performed.
The code also needs reworking to support Rust idioms, including borrows, optionals, non-pointer arithmetic, and iterators.
However, the function `tinfl_decompress` needs significant reworking, since it relies on `goto` spaghetti.

All the loops created from C `while` and `for` loops must be reviewed for bounds.
This also needs general stress testing.

## License
miniz-rs is licensed under the same terms as the Rust distribution, for which it is intended.
