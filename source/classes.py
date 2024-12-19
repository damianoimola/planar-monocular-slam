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
