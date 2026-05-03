"""Compatibility entry point for ACT training.

The original ACT examples use imitate_episodes.py. This project keeps the
training implementation in train.py, so this wrapper preserves the familiar
command shape without duplicating training logic.
"""

from train import main


if __name__ == '__main__':
    main()
