# Contributing

1. Keep model weights, generated media, runtime caches and credentials out of Git.
2. Run `./doctor.sh`, then the focused unit tests for the code you changed.
3. For scheduler or numerical changes, include an exact comparator and explain
   whether the change affects speed, memory, numerical output, or video quality.
4. Do not broaden hardware claims beyond tested evidence. RTX 4090/SM89 is the
   calibrated platform; other devices require their own validation.
5. Preserve FL2VA/Ref2VA task-family separation and LoRA profile metadata.

The project currently uses an all-rights-reserved public-source license.
Contributions require explicit acceptance by the repository owner.
