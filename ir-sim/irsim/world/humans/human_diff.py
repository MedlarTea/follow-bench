from irsim.world import ObjectBase

class HumanDiff(ObjectBase):
    def __init__(self, color="b", **kwargs):
        super(HumanDiff, self).__init__(color=color, role="human", **kwargs
        )
