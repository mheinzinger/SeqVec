# Changelog

## Unreleased

 * Instead of `--protein 1` it's now simply `--protein`
 * Automatic CPU fallback for sequences to long for the GPU

## 0.3.0 - 2020-05-14

 * Changed `--no-sum-layers` to `--layer` with options `sum`, `all`, `CNN`, `LSTM1` and `LSTM2`. The default remains summing up all three layer.