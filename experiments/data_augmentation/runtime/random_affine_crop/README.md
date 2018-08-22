Performed data augmentation using torchvision's RandomAffine and RandomResizeCrop transforms

RandomAffine performs skew, rotation, and shear.

RandomResizeCrop essentially crops out a randomly sized rectangle, then resizes it to the correct shape.

Result was a 9-10 point increase in all models!
