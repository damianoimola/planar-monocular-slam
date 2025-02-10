class Trajectory:
    ids = []
    odoms = []
    gts = []

    def __len__(self):
        return len(self.ids)

    def add(self, id, odom, gt):
        self.ids.append(id)
        self.odoms.append(odom)
        self.gts.append(gt)

    def get_odom(self, id):
        return self.odoms[id]

    def get_gt(self, id):
        return self.gts[id]


class SinglePoint:
    sequential_number = None
    points_id_actual = None
    point = None

    def __init__(self, sequential_number, points_id_actual, point):
        self.sequential_number = int(sequential_number)
        self.points_id_actual = int(points_id_actual)
        self.point = (float(point[0]), float(point[1]))

    def get_id_and_point(self):
        return self.points_id_actual, self.point

class Measurement:
    seq = None
    odom = None
    gt = None
    points = None

    def __init__(self, seq, odom, gt, points):
        self.seq = seq
        self.odom = odom
        self.gt = gt
        self.points = points

    def __len__(self):
        return len(self.points)

    def get_points(self):
        return self.points

    def get_ids_and_points(self):
        temp_list = [point.get_id_and_point() for point in self.points]

        ids = [item[0] for item in temp_list]
        points = [item[1] for item in temp_list]

        return ids, points