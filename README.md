miniz-rs
========

A translation of the miniz.c single source Deflate/Inflate compression library to Rust.

Presently, Rust's libflate relies on miniz.c. Other compression libraries like [flate2-rs](https://github.com/alexcrichton/flate2-rs) rely on a statically linked miniz. Having a pure-Rust compression implementation would simplify the implementation of all compression libraries.

## Progress

Basic translations such as function signatures, type declarations, C-only operator replacements, type checking, mutability, lifetimes and visibility have been done. However, the function `tinfl_decompress` needs significant reworking, since it relies on `goto` spaghetti.

All the loops created from C `while` and `for` loops and every former increment or decrement operator needs to be reviewed.

miniz-rs desperately needs input testing, to verify that bugs weren't introduced during translation.

## License
miniz-rs is licensed under the same terms as the Rust distribution, for which it is intended.
