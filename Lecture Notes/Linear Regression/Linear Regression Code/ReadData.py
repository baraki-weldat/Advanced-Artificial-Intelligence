import numpy as np



def get_house_data(file_name="house.csv"):

    data = []

    def parse_line(l):
        l = l.strip().split(",")
        return [int(x) for x in l]



    with open(file_name) as f:

        first = True
        for line in f:

            if first:
                first = False
                continue

            r = parse_line(line)
            data.append(r)


    return np.asarray(data)
