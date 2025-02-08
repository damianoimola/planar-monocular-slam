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
    points_id_meas = None
    points_id_actual = None
    point = None

    def __init__(self, points_id_meas, points_id_actual, point):
        self.points_id_meas = int(points_id_meas)
        self.points_id_actual = int(points_id_actual)
        self.point = (float(point[0]), float(point[1]))

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

    def get_point(self, seq):
        return self.points[seq]

    def get_points(self):
        return self.points

    # to make it iterable
    def __iter__(self):
        for p in self.points:
            yield p.points_id_actual, p.point
            # yield p.points_id_meas, p.point