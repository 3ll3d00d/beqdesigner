import logging

import numpy as np
import struct
import os

logger = logging.getLogger('madvr')


def parse_measurements(file):
    '''
    parses a measurements file into data using the following format

    DWORD magic;   // "mvr+"
    DWORD version;  // 4 (or higher)
    DWORD headerSize;  // 7 (or more, in DWORDs)
    DWORD sceneCount;
    DWORD videoDuration;  // in frames
    DWORD flags;  // 0x00000001 = complete
    DWORD maxCLL;  // measured MaxCLL

    Scene Data
    Starts at byte "headerSize * sizeof(DWORD)".
    has 3 segments:
        1) The first frame of the scenes
        2) The last frame of the scenes
        3) The peak nits of the scenes.

    So if e.g. sceneCount says 5, then there are 5 scenes.
    The file will have
    - 5 DWORDs giving you the frame number of the first frame in each scene.
    - 5 DWORDs giving you the frame number (+1) of the last frame in each scene.
    - 5 DWORDs giving you the measured peak of each frame, in Nits.

    Finally, at file offset "headerSize * sizeof(DWORD) + 3 * sceneCount * sizeof(DWORD)", there will be detailed measurements for each video frame.

    The "videoDuration" will define how many of those detailed measurements are in the file.

    For each frame there are 32 WORDs stored.

        1) The first WORD is the measured peak of the frame, in PQ light.
        You need to divide that number by 64000.0, and then decode the PQ transfer function, and then multiply with 10000.0 to get the Nits number.
        2) The other remaining 31 WORDs have the percentage values of each of the 31 histogram bars.
        Divide by 640.0 to get a percentage number from 0.0 - 100.0.


    static double pq2linear(double pq)
    {
      double temp = pow(pq, 1.0 / 78.84375);
      return pow(max(temp - 0.8359375, 0) / (18.8515625 - 18.6875 * temp), 1.0 / 0.1593017578125);
    }

    This function expects the "pq" value to be in the 0.0-1.0 range, and the linear return value will also be in the same range.
    You have to multiply the linear return value with 10000.0 to get the nits number.

    :param file: the full path to a file.
    :return HDRData
    '''
    with open(file, "rb") as f:
        f.seek(4, 1)
        version = struct.unpack('i', f.read(4))[0]
        header_size = struct.unpack('i', f.read(4))[0]
        scene_count = struct.unpack('i', f.read(4))[0]
        frame_count = struct.unpack('i', f.read(4))[0]
        flags = struct.unpack('i', f.read(4))[0]
        max_cll = struct.unpack('i', f.read(4))[0]
        logger.info(f"Imported header from {file}")
        logger.info(f"Scene Count: {scene_count}")
        logger.info(f"Frame Count: {frame_count}")
        logger.info(f"Flags: {flags}")
        logger.info(f"MaxCLL: {max_cll}")
        logger.info(f"File Position: {f.tell()}")
        if flags != 1:
            logger.warning(f"{file} is incomplete")
        else:
            scene_start = np.frombuffer(f.read(4 * scene_count), dtype=np.uint32)
            logger.info(f"File Position: {f.tell()}")
            scene_end = np.frombuffer(f.read(4 * scene_count), dtype=np.uint32)
            logger.info(f"File Position: {f.tell()}")
            peak_nits = np.frombuffer(f.read(4 * scene_count), dtype=np.uint32)
            logger.info(f"File Position: {f.tell()}")
            pq_histogram = np.frombuffer(f.read(2 * frame_count * 32), dtype=np.uint16)
            logger.info(f"File Position: {f.tell()}")
            logger.info(f"File Size: {os.path.getsize(file)}")
            hdr_data = HDRData(file,
                               max_cll,
                               np.column_stack((scene_start, scene_end, peak_nits)),
                               pq_histogram.reshape((-1, 32)))
            return hdr_data


class HDRData:
    def __init__(self, file, maxcll, scenes, pq_histogram):
        self.__file = file
        self.__maxcll = maxcll
        self.__scenes = scenes
        self.__pq_histogram = pq_histogram
        # double temp = pow(pq, 1.0 / 78.84375);
        # return pow(max(temp - 0.8359375, 0) / (18.8515625 - 18.6875 * temp), 1.0 / 0.1593017578125);
        tmp1 = pq_histogram[:, 0] / 64000.0
        tmp2 = np.power(tmp1, 1.0 / 78.84375)
        tmp3 = tmp2 - 0.8359375
        tmp3[tmp3 < 0] = 0
        tmp4 = 18.8515625 - (18.6875 * tmp2)
        tmp5 = tmp3 / tmp4
        self.__peak_nits = np.power(tmp5, 1.0 / 0.1593017578125) * 10000.0
        self.__histogram = pq_histogram[:, 1:] / 640.0

    @property
    def file(self):
        return self.__file

    @property
    def maxcll(self):
        return self.__maxcll

    @property
    def scenes(self):
        return self.__scenes

    @property
    def peak_nits(self):
        return self.__peak_nits

    @property
    def histogram(self):
        return self.__histogram


if __name__ == '__main__':
    hdr_data = parse_measurements('d:/junk/index.bdmv.measurements')
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ims = ax1.imshow(np.transpose(hdr_data.histogram), aspect='auto')
    ax2.plot(hdr_data.peak_nits)
    plt.tight_layout(h_pad=0)
    plt.show()
