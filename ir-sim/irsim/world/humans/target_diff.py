from irsim.world import ObjectBase

class TargetDiff(ObjectBase):
    def __init__(self, color="orange", **kwargs):
        super(TargetDiff, self).__init__(color=color, role="target", **kwargs
        )
