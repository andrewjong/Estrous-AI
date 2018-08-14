ADDS TO THE RANDOMAFFINE RANDOMRESIZECROP EXPERIMENT!

Added the following line to transforms to perform color jittering:

```python
# ColorJitter values chosen somewhat arbitrarily by what "looked" good
# possibly something to optimize
transforms.ColorJitter(brightness=0.20, saturation=0.70, contrast=0.5, hue=0.10),
```

Line must be uncommented or commented to use/not-use.

## Results
Found ColorJitter improves diestrus-vs-all by 2-3 points, degrades 4-class by 1-2 points, and 3-class (without transfer learning from other Estrous-AI models) remains about the same.
