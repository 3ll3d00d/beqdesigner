'''
Builders for the parts of a DVD-Video disc's IFO files that model.dvd reads -- enough for the title table and the
program chain durations, laid out as libdvdread's ifo_types.h describes. Not a valid disc (no VOBs or navigation
tables), so nothing here can be *played*; it exists to test title selection.
'''
import os
import struct

SECTOR = 2048


def _bcd(n):
    return ((n // 10) << 4) | (n % 10)


def dvd_time(hours=0, minutes=0, seconds=0, frames=0, fps=25):
    ''' A dvd_time_t: BCD h/m/s and a frame count whose top two bits give the rate (01 = 25 fps, 11 = 29.97). '''
    rate_bits = {25: 1, 30: 3}[fps]
    return bytes([_bcd(hours), _bcd(minutes), _bcd(seconds), (rate_bits << 6) | _bcd(frames)])


def vmg_ifo(titles, magic=b'DVDVIDEO-VMG'):
    '''
    :param titles: (title_set_nr, vts_ttn, chapters[, angles]) per title, in title-number order.
    '''
    data = bytearray(SECTOR * 2)
    data[:12] = magic
    struct.pack_into('>I', data, 0xC4, 1)  # TT_SRPT starts at sector 1
    body = bytearray()
    for title in titles:
        vts, vts_ttn, chapters = title[:3]
        angles = title[3] if len(title) > 3 else 1
        body += struct.pack('>BBHHBBI', 0, angles, chapters, 0, vts, vts_ttn, 0)
    struct.pack_into('>HHI', data, SECTOR, len(titles), 0, 7 + len(body))
    data[SECTOR + 8:SECTOR + 8 + len(body)] = body
    return bytes(data)


def vts_ifo(title_pgcs, pgc_times, magic=b'DVDVIDEO-VTS'):
    '''
    :param title_pgcs: for each title of the set (VTS_TTN 1..n), its chapters as (pgcn, pgn) pairs.
    :param pgc_times: a dvd_time() per program chain, in PGC-number order (1-based).
    '''
    data = bytearray(SECTOR * 3)
    data[:12] = magic
    struct.pack_into('>I', data, 0xC8, 1)  # VTS_PTT_SRPT at sector 1
    struct.pack_into('>I', data, 0xCC, 2)  # VTS_PGCIT at sector 2

    offsets, lists = [], b''
    base = 8 + 4 * len(title_pgcs)
    for chapters in title_pgcs:
        offsets.append(base + len(lists))
        lists += b''.join(struct.pack('>HH', pgcn, pgn) for pgcn, pgn in chapters)
    ptt = struct.pack('>HHI', len(title_pgcs), 0, base + len(lists) - 1)
    ptt += b''.join(struct.pack('>I', o) for o in offsets) + lists
    data[SECTOR:SECTOR + len(ptt)] = ptt

    pgc_base = 8 + 8 * len(pgc_times)
    entries, blocks = b'', b''
    for playback in pgc_times:
        entries += struct.pack('>II', 0x80000000, pgc_base + len(blocks))
        blocks += struct.pack('>HBB', 0, 1, 1) + playback + bytes(8)  # two zero bytes, programs, cells, time, pad
    pgcit = struct.pack('>HHI', len(pgc_times), 0, pgc_base + len(blocks) - 1) + entries + blocks
    data[2 * SECTOR:2 * SECTOR + len(pgcit)] = pgcit
    return bytes(data)


def write_disc(root, titles, title_sets, video_ts='VIDEO_TS'):
    '''
    :param titles: as vmg_ifo().
    :param title_sets: {vts_number: (title_pgcs, pgc_times)} as vts_ifo().
    :return: the disc root.
    '''
    folder = os.path.join(str(root), video_ts)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'VIDEO_TS.IFO'), 'wb') as f:
        f.write(vmg_ifo(titles))
    for number, (title_pgcs, pgc_times) in title_sets.items():
        with open(os.path.join(folder, f"VTS_{number:02d}_0.IFO"), 'wb') as f:
            f.write(vts_ifo(title_pgcs, pgc_times))
    return str(root)
