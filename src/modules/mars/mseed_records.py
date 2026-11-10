"""Minimal miniSEED reader (Steim1/Steim2/int16/int32/float32), dependency-free."""
import struct
import numpy as np
from datetime import datetime, timedelta

def _steim_words(rec, off, nframes):
    return struct.unpack('>%dI' % (16 * nframes), rec[off:off + 64 * nframes])

def _sign(v, bits):
    return v - (1 << bits) if v & (1 << (bits - 1)) else v

def decode_steim(rec, off, nsamp, enc, nframes):
    w = _steim_words(rec, off, nframes)
    diffs = []
    x0 = xn = None
    for f in range(nframes):
        fw = w[16 * f:16 * f + 16]
        nibs = [(fw[0] >> (30 - 2 * i)) & 3 for i in range(16)]
        for i in range(1, 16):
            if f == 0 and i == 1: x0 = _sign(fw[1], 32); continue
            if f == 0 and i == 2: xn = _sign(fw[2], 32); continue
            n = nibs[i]; word = fw[i]
            if enc == 10:
                if n == 1:   diffs += [_sign((word >> s) & 0xFF, 8) for s in (24, 16, 8, 0)]
                elif n == 2: diffs += [_sign((word >> s) & 0xFFFF, 16) for s in (16, 0)]
                elif n == 3: diffs += [_sign(word, 32)]
            else:
                if n == 1: diffs += [_sign((word >> s) & 0xFF, 8) for s in (24, 16, 8, 0)]
                elif n in (2, 3):
                    dnib = word >> 30
                    if n == 2:
                        if dnib == 1: diffs += [_sign(word & 0x3FFFFFFF, 30)]
                        elif dnib == 2: diffs += [_sign((word >> s) & 0x7FFF, 15) for s in (15, 0)]
                        elif dnib == 3: diffs += [_sign((word >> s) & 0x3FF, 10) for s in (20, 10, 0)]
                    else:
                        if dnib == 0: diffs += [_sign((word >> s) & 0x3F, 6) for s in (24, 18, 12, 6, 0)]
                        elif dnib == 1: diffs += [_sign((word >> s) & 0x1F, 5) for s in (25, 20, 15, 10, 5, 0)]
                        elif dnib == 2: diffs += [_sign((word >> s) & 0xF, 4) for s in (24, 20, 16, 12, 8, 4, 0)]
    diffs = np.array(diffs[:nsamp], dtype=np.int64)
    out = np.empty(len(diffs), dtype=np.int64)
    out[0] = x0
    # first diff is x0 - x_prev (unused); samples are x0 + cumsum(diffs[1:])
    out[1:] = x0 + np.cumsum(diffs[1:])
    return out, xn

def read_mseed(path):
    b = open(path, 'rb').read()
    # detect record length from blockette 1000 of first record
    recs = []
    pos = 0
    reclen = None
    while pos + 48 <= len(b):
        h = b[pos:pos + 48]
        if h[:6] == b'\x00' * 6: break
        yr, doy, hh, mm, ss, _, fr = struct.unpack('>HHBBBBH', h[20:30])
        ns, sf, sm = struct.unpack('>Hhh', h[30:36])
        doff, boff = struct.unpack('>HH', h[44:48])
        enc = rl = None
        while boff:
            bt, nxt = struct.unpack('>HH', b[pos + boff:pos + boff + 4])
            if bt == 1000:
                enc = b[pos + boff + 4]; rl = 1 << b[pos + boff + 6]
            boff = nxt
        reclen = rl or reclen
        t0 = datetime(yr, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss, microseconds=fr * 100)
        fs = sf * sm if sf > 0 and sm > 0 else (-sf / sm if sf < 0 and sm > 0 else sf / -sm if sm < 0 and sf > 0 else 1.0)
        if enc in (10, 11):
            d, _ = decode_steim(b[pos:pos + reclen], doff, ns, enc, (reclen - doff) // 64)
        elif enc == 1:
            d = np.frombuffer(b[pos + doff:pos + doff + 2 * ns], '>i2').astype(np.int64)
        elif enc == 3:
            d = np.frombuffer(b[pos + doff:pos + doff + 4 * ns], '>i4').astype(np.int64)
        elif enc == 4:
            d = np.frombuffer(b[pos + doff:pos + doff + 4 * ns], '>f4').astype(np.float64)
        elif enc == 5:
            d = np.frombuffer(b[pos + doff:pos + doff + 8 * ns], '>f8').astype(np.float64)
        else:
            raise ValueError(f'unsupported encoding {enc}')
        recs.append((t0, fs, ns, enc, np.asarray(d)))
        pos += reclen
    return recs