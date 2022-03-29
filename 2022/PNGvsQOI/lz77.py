import re
from sre_constants import ANY_ALL

from numpy import mat


def lz77(string):
    output = []
    i = 0
    for index, c in enumerate(string):
        search_buffer: str = string[0:i]

        # we finished
        if i > len(string) - 1:
            break

        if search_buffer == "":
            o = 0
            l = 0
            c = string[i]

        else:
            stop = False
            lookahead = 1
            last_match_start_index = 0
            while not stop and (i + lookahead) < len(string):
                match_start_index = search_buffer.find(string[i : i + lookahead])

                if match_start_index == -1:
                    stop = True
                else:
                    lookahead += 1
                    last_match_start_index = i - match_start_index

            if lookahead == 1:
                o = 0
                l = 0
                c = string[i]
            else:
                o = last_match_start_index
                l = lookahead - 1
                c = string[i + lookahead - 1]

        i += l + 1

        output.append((o, l, c))

    return output


if __name__ == "__main__":

    string = "ababcbababaa"

    output = lz77(string=string)
    print(output)
