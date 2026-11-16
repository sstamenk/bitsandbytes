Speedup in total time over the 40 shapes, this branch vs main (see plots_vs_main.py)

| GPU | dtype | M=1 | M=4 | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 | M=384 |
|---|---|---|---|---|---|---|---|---|---|---|
| Radeon Pro W7900 (gfx1100) | fp16 | 1.07x | 1.83x | 2.87x | 5.23x | 3.67x | 2.38x | 1.62x | 1.16x | 1.02x |
| Radeon Pro W7900 (gfx1100) | bf16 | 1.06x | 1.79x | 2.80x | 5.76x | 3.99x | 2.54x | 1.63x | 1.16x | 1.05x |
| Radeon Pro W7900 (gfx1100) | fp32 | 1.00x | 1.00x | 1.63x | 4.35x | 2.54x | 1.40x | 1.05x | 1.00x | 1.00x |
| Radeon AI PRO R9700 (gfx1201) | fp16 | 1.00x | 1.65x | 2.72x | 5.86x | 4.67x | 2.97x | 1.96x | 1.28x | 1.32x |
| Radeon AI PRO R9700 (gfx1201) | bf16 | 1.00x | 1.62x | 2.66x | 5.85x | 4.63x | 2.99x | 1.96x | 1.28x | 1.35x |
| Radeon AI PRO R9700 (gfx1201) | fp32 | 1.00x | 1.00x | 1.54x | 6.04x | 3.85x | 2.34x | 1.74x | 1.45x | 1.36x |
| Radeon 8060S (gfx1151) | fp16 | 1.11x | 1.61x | 2.58x | 6.50x | 5.54x | 3.92x | 2.98x | 2.20x | 1.86x |
| Radeon 8060S (gfx1151) | bf16 | 1.09x | 1.63x | 3.00x | 7.06x | 5.87x | 4.08x | 2.83x | 1.99x | 1.82x |
| Radeon 8060S (gfx1151) | fp32 | 1.00x | 1.22x | 2.12x | 17.81x | 10.28x | 6.62x | 7.71x | 8.19x | 8.72x |
| Radeon 760M (gfx1103) | fp16 | 1.01x | 1.73x | 2.50x | 8.63x | 7.47x | 4.89x | 2.78x | 1.70x | - |
| Radeon 760M (gfx1103) | bf16 | 1.02x | 1.65x | 2.34x | 8.59x | 7.40x | 4.79x | 2.71x | 1.64x | - |
| Radeon 760M (gfx1103) | fp32 | 1.00x | 0.97x | 2.05x | 9.20x | 5.33x | 3.85x | 3.49x | 3.31x | - |
