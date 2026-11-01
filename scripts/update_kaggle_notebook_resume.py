import json

nb_path = 'notebooks/train_lumacam_kaggle.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'for epoch in range(1, EPOCHS + 1):' in src:
            resume_block = (
                "# Checkpoint Resuming (allows splitting training across multiple Kaggle sessions/accounts)\n"
                "RESUME_CHECKPOINT = None  # e.g. Path('/kaggle/input/lumacam-checkpoint/lumacam_best.pt')\n"
                "start_epoch = 1\n"
                "if RESUME_CHECKPOINT and Path(RESUME_CHECKPOINT).exists():\n"
                "    print(f'[Resume] Loading checkpoint from {RESUME_CHECKPOINT}')\n"
                "    ckpt = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)\n"
                "    generator.load_state_dict(ckpt.get('generator_state_dict', ckpt), strict=False)\n"
                "    if 'discriminator_state_dict' in ckpt:\n"
                "        discriminator.load_state_dict(ckpt['discriminator_state_dict'], strict=False)\n"
                "    start_epoch = ckpt.get('epoch', 0) + 1\n"
                "    best_psnr = ckpt.get('val_psnr', -1.0)\n"
                "    print(f'[Resume] Resumed from Epoch {start_epoch} (Val PSNR: {best_psnr:.2f} dB)!')\n\n"
            )
            src = src.replace('for epoch in range(1, EPOCHS + 1):', resume_block + 'for epoch in range(start_epoch, EPOCHS + 1):')
            cell['source'] = [line + '\n' for line in src.splitlines()]
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully with resume capability!')
