# FPRange Domain

An element of the FPRange domain has the form:

```cpp
struct {
    lo:      fp16  // lower bound
    hi:      fp16  // upper bound
    has_nan: bool  // whether NaN is included
}
```

## Interpretation

The element is interpreted based on whether `lo` and `hi` are valid (non-NaN) and ordered. 
Note that `lo` and `hi` can be `+inf/-inf` and the comparsion is well-defined on them.

```cpp
if (!is_nan(lo) && !is_nan(hi) && lo <= hi) {
    if (!has_nan)
        // Normal Interval
    else
        // Normal Interval with NaN
} else {
    if (!has_nan)
        // Bottom
    else
        // NaN Only
}
```

- **Normal Interval**: $\{x \mid lo \le x \le hi\}$
- **Normal Interval with NaN**: $\{x \mid lo \le x \le hi\} \cup \{\mathrm{NaN}\}$
- **NaN Only**: $\{\mathrm{NaN}\}$
- **Bottom**: $\varnothing$
