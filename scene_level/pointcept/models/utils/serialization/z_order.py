# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:
    # Key Look-Up Table
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                # all depth-8 bits for x -> key
                self.xyz2key(r256, zero, zero, 8),
                # all depth-8 bits for y -> key
                self.xyz2key(zero, r256, zero, 8),
                # all depth-8 bits for z -> key
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        # all depth-9 keys -> xyz
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        # map i-th bit of xyz to (i*3 + 0/1/2) -th bit of keys
        # => [..., (i*3)+2 : (i*3)+0, ...] bits of key = cat[i-th bits of xyz]
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        # reside (3*i + 0/1/2)-th bit back to each i-th bit
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    # [8**2=256] - pre-computed bit mapping for x/y/z -> key (position in bits)
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    # all 1s bit mask up to depth-8
    mask = 255 if depth > 8 else (1 << depth) - 1
    # map bits into key
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        # proc the higher 8-bits - assume depth <= 16
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        # prepend batch bits - b.bit_length() <= 16, as using long (at most 64-bits for int64)
        b = b.long()
        key = b << 48 | key

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    # [2**9=512] - pre-computed bit mapping for key -> x/y/z (position in bits)
    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    # mask out the lower 48 (16*3) bits of key - mapped from x/y/z
    key = key & ((1 << 48) - 1)

    # #iter for mapping with DX/DY/DZ - 9-bits key maps into 3-bits x/y/z
    n = (depth + 2) // 3
    for i in range(n):
        # k = i-th lower 9-bits in key
        k = key >> (i * 9) & 511
        # - maps to i-th lower 3-bits in x/y/z
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b
