"""Bounded feedback controller. Test outcomes are not an accepted input."""
import math

def choose_capacity(feedback=None):
    if feedback is None:return 16,'initial compact network'
    if set(feedback)!={'train_loss','val_loss'}:
        raise ValueError('Controller accepts training/validation loss only')
    train=float(feedback['train_loss']);val=float(feedback['val_loss'])
    if not math.isfinite(train) or not math.isfinite(val) or min(train,val)<0:
        raise ValueError('Invalid feedback')
    if train+.05<val:return 8,'reduce capacity after generalization gap'
    return 24,'test greater capacity after feedback'
