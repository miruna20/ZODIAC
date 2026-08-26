# ZODIAC checkpoints

Trained VAE + diffusion checkpoints are **not** stored in the repository. Download them from LRZ Sync+Share
and unzip them here.

## Download & unzip

Checkpoints archive: https://syncandshare.lrz.de/getlink/fiWdgq64ZePMA6GVaXvzJ6/

```
checkpoints/
├── zodiac/                     # ZODIAC / ZODIAC-clean prior (trained on deformed + TotalSegmentator)
│   ├── diffusion/df_steps-latest.pth   # HR diffusion prior 
│   └── vae/vae_steps-latest.pth        # VAE 
└── tp-odiac/                   # TP-ODIAC (supervised) checkpoints
    ├── diffusion/df_steps-latest.pth   # HR diffusion prior 
    └── vae/vae_steps-latest.pth        # VAE
```

ZODIAC and ZODIAC-clean share the same `zodiac/` prior + VAE and differ only by the
inference flag (`use_partial_pcd_zero_shot` vs `use_partial_pcd_zero_shot_clean`).
