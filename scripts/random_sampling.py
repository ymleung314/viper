"""
Uniformly random-picks N_VEC from INPUT_FILE where each vector has a dimension of VEC_DIM.
INPUT_FILE is in the format specified in DiskANN.
Results are saved to OUTPUT_FILE in the same format.
"""

import os
import random
import struct

from tqdm import tqdm

INPUT_FILE = ""
OUTPUT_FILE = ""
N_VEC = 1000000
VEC_DIM = 384

FLOAT_SIZE = 4


def main():
    if INPUT_FILE == "" or OUTPUT_FILE == "":
        print("Input/output file (INPUT_FILE/OUTPUT_FILE) not specified")
        return

    inf_size = os.stat(INPUT_FILE).st_size
    payload_size = inf_size - 8
    total = payload_size / VEC_DIM / FLOAT_SIZE

    print(f"Vector dimension: {VEC_DIM}")
    print(f"Vector element size: {FLOAT_SIZE}")
    print(f"Total count: {total}")
    print(f"Subset count: {N_VEC}")

    with (
        tqdm(total=total) as pbar,
        open(INPUT_FILE, "rb") as inf,
        open(OUTPUT_FILE, "wb") as outf,
    ):
        # Header.
        outf.write(struct.pack("<ii", N_VEC, VEC_DIM))

        # Skip input header.
        HEADER_SIZE = 8
        inf.seek(HEADER_SIZE)

        # Reservoir sampling.
        i = 0
        while vec := inf.read(VEC_DIM * FLOAT_SIZE):
            if i < N_VEC:
                outf.write(vec)
            else:
                target = random.randrange(i + 1)
                if target < N_VEC:
                    outf.seek(HEADER_SIZE + target * VEC_DIM * FLOAT_SIZE)

            i += 1
            pbar.update(1)


if __name__ == "__main__":
    main()
