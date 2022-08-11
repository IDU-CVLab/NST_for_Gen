# NST_for_Gen
### Example testing command:
    $ python test.py --content "input/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t03_ch00.png" --style "input/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t03_ch00.tif" --style_mask "input/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t03_ch00.png"

### Example training command:
    $ python train.py --content_dir "/path/to/masks" --style_dir "/path/to/data"

### Example evaluation data generation command:
    $ python eval.py --content_dir "/path/to/masks" --style_dir "/path/to/data"
